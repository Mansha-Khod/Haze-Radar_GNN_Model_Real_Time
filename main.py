import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from contextlib import asynccontextmanager
import aiohttp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "temperature", "humidity", "wind_speed", "wind_direction",
    "avg_fire_confidence", "upwind_fire_count", "population_density"
]

# Extended features include temporal encoding AND current PM2.5 for persistence
EXTENDED_FEATURE_COLS = FEATURE_COLS + ["hour_sin", "hour_cos", "is_rush_hour", "current_pm25"]

FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/realtime_haze_gnn_infer.pt")
NORM_STATS_PATH = os.getenv("NORM_STATS_PATH", "artifacts/normalization_stats.json")
GRAPH_PATH = os.getenv("GRAPH_PATH", "artifacts/city_graph.json")

CITY_COORDINATES = {
    "Bekasi": {"lat": -6.2383, "lon": 106.9756},
    "Karawang": {"lat": -6.3063, "lon": 107.3019},
    "Sumedang": {"lat": -6.8575, "lon": 107.9167},
    "Tasikmalaya": {"lat": -7.3274, "lon": 108.2207},
    "Bandung": {"lat": -6.9175, "lon": 107.6191},
    "Subang": {"lat": -6.5697, "lon": 107.7631},
    "Indramayu": {"lat": -6.3269, "lon": 108.3200},
    "Cimahi": {"lat": -6.8722, "lon": 107.5425},
    "West Bandung": {"lat": -6.8597, "lon": 107.4858},
    "Cianjur": {"lat": -6.8167, "lon": 107.1392}
}


def pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 to US AQI"""
    if pm25 < 0:
        return 0
    elif pm25 <= 12.0:
        return int(pm25 * 50 / 12.0)
    elif pm25 <= 35.4:
        return int(50 + (pm25 - 12.0) * 50 / 23.4)
    elif pm25 <= 55.4:
        return int(100 + (pm25 - 35.4) * 50 / 20.0)
    elif pm25 <= 150.4:
        return int(150 + (pm25 - 55.4) * 100 / 95.0)
    elif pm25 <= 250.4:
        return int(200 + (pm25 - 150.4) * 100 / 100.0)
    elif pm25 <= 350.4:
        return int(300 + (pm25 - 250.4) * 100 / 100.0)
    else:
        return int(400 + (pm25 - 350.4) * 100 / 149.6)


def aqi_to_category(aqi: float) -> str:
    """Convert AQI to health category"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


async def fetch_weather_forecast_batch() -> Dict[str, List[Dict]]:
    """Fetch 72-hour weather forecasts for all cities"""
    async def fetch_one_city(city: str, coords: Dict) -> tuple:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "forecast_days": 3,
            "timezone": "auto"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        hourly = data.get("hourly", {})
                        temps = hourly.get("temperature_2m", [])
                        humidity = hourly.get("relative_humidity_2m", [])
                        wind_speed = hourly.get("wind_speed_10m", [])
                        wind_dir = hourly.get("wind_direction_10m", [])
                        
                        forecasts = []
                        for h in range(min(72, len(temps))):
                            forecasts.append({
                                "hour": h,
                                "temperature": temps[h] if h < len(temps) else 26.0,
                                "humidity": humidity[h] if h < len(humidity) else 80.0,
                                "wind_speed": wind_speed[h] if h < len(wind_speed) else 3.0,
                                "wind_direction": wind_dir[h] if h < len(wind_dir) else 180.0
                            })
                        return (city, forecasts)
        except Exception as e:
            logger.error(f"Weather fetch failed for {city}: {e}")
        
        # Fallback
        return (city, [{
            "hour": h, 
            "temperature": 26.0, 
            "humidity": 80.0, 
            "wind_speed": 3.0,
            "wind_direction": 180.0
        } for h in range(72)])
    
    tasks = [fetch_one_city(city, coords) for city, coords in CITY_COORDINATES.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)


class HazeForecastGNN(nn.Module):
    """Graph Neural Network for PM2.5 forecasting"""
    
    def __init__(self, in_features: int, hidden_dim: int = 128, 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(self.ln1(h))
        h = self.dropout(h)
        
        h2 = self.gat2(h, edge_index)
        h2 = self.ln2(h2)
        h = F.elu(h + h2)
        
        pm25_pred = self.prediction_head(h)
        uncertainty = self.uncertainty_head(h)
        
        return pm25_pred, uncertainty


class ModelManager:
    """Manages GNN model and predictions"""
    
    def __init__(self):
        self.model = None
        self.graph = None
        self.edge_index = None
        self.city_to_idx = None
        self.cities = None
        self.norm_stats = None
        self.weather_cache = {}
        self.current_predictions = []
        self.city_forecasts = {}
        self.supabase = None
        
        if SUPABASE_URL and SUPABASE_KEY:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def load_graph(self):
        """Load city graph structure"""
        logger.info("Loading graph structure...")
        with open(GRAPH_PATH, 'r') as f:
            self.graph = json.load(f)
        
        self.city_to_idx = self.graph["city_to_idx"]
        self.cities = sorted(self.city_to_idx.keys(), key=lambda c: self.city_to_idx[c])
        
        edges = self.graph["edges"]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        logger.info(f"Graph loaded: {len(self.cities)} cities, {len(edges)} edges")
    
    def load_model(self):
        """Load trained GNN model"""
        logger.info("Loading model...")
        
        # Load normalization stats
        with open(NORM_STATS_PATH, 'r') as f:
            self.norm_stats = json.load(f)
        
        # Determine number of features from normalization stats
        num_features = len(self.norm_stats["feature_mean"])
        
        # Initialize model
        self.model = HazeForecastGNN(
            in_features=num_features,
            hidden_dim=128,
            num_heads=4,
            dropout=0.2
        )
        
        # Load weights
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"Model loaded successfully (features: {num_features})")
    
    def fetch_current_data(self) -> Dict:
        """Fetch latest data from Supabase"""
        if not self.supabase:
            return {}
        
        try:
            response = self.supabase.table("gnn_training_data")\
                .select("city," + ",".join(FEATURE_COLS + ["current_pm25", "current_aqi", "created_at"]))\
                .order("created_at", desc=True)\
                .limit(100)\
                .execute()
            
            city_data = {}
            seen = set()
            
            for row in response.data:
                city = row.get("city")
                if not city or city in seen:
                    continue
                
                seen.add(city)
                city_data[city] = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
                city_data[city]["current_pm25"] = float(row.get("current_pm25", 0))
                city_data[city]["current_aqi"] = float(row.get("current_aqi", 0))
                city_data[city]["timestamp"] = row.get("created_at", "")
            
            logger.info(f"Fetched data for {len(city_data)} cities from Supabase")
            return city_data
            
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return {}
    
    def build_feature_matrix(self, city_data: Dict, hour_offset: int = 0) -> np.ndarray:
        """
        Build feature matrix for all cities at a given time offset.
        
        Includes temporal features to handle day/night patterns.
        Uses current PM2.5 as anchor for stability.
        """
        feature_matrix = np.zeros((len(self.cities), len(EXTENDED_FEATURE_COLS)), dtype=np.float32)
        
        # Calculate the predicted hour
        predicted_hour = (datetime.now() + timedelta(hours=hour_offset)).hour
        
        # Temporal features (same for all cities at this hour)
        hour_sin = np.sin(2 * np.pi * predicted_hour / 24)
        hour_cos = np.cos(2 * np.pi * predicted_hour / 24)
        is_rush_hour = 1.0 if (7 <= predicted_hour <= 9 or 17 <= predicted_hour <= 19) else 0.0
        
        for i, city in enumerate(self.cities):
            if city not in city_data:
                # Use defaults if no data
                pm25_baseline = 30.0  # Default PM2.5
                feature_matrix[i] = [26.0, 80.0, 3.0, 180.0, 0.0, 0.0, 5000.0, 
                                    hour_sin, hour_cos, is_rush_hour, pm25_baseline]
                continue
            
            # Get weather forecast for this hour
            weather_list = self.weather_cache.get(city, [])
            if hour_offset < len(weather_list):
                weather = weather_list[hour_offset]
            else:
                weather = {"temperature": 26.0, "humidity": 80.0, 
                          "wind_speed": 3.0, "wind_direction": 180.0}
            
            # Build feature vector
            current_data = city_data[city]
            
            # CRITICAL: Always use CURRENT actual PM2.5 as baseline
            # Model will learn to predict small deviations from this
            pm25_baseline = current_data.get("current_pm25", 30.0)
            
            feature_matrix[i] = [
                weather.get("temperature", 26.0),
                weather.get("humidity", 80.0),
                weather.get("wind_speed", 3.0),
                weather.get("wind_direction", 180.0),
                current_data.get("avg_fire_confidence", 0.0),
                current_data.get("upwind_fire_count", 0.0),
                current_data.get("population_density", 5000.0),
                hour_sin,
                hour_cos,
                is_rush_hour,
                pm25_baseline  # ALWAYS current PM2.5, not previous prediction
            ]
        
        return feature_matrix
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using saved statistics"""
        mean = np.array(self.norm_stats["feature_mean"])
        std = np.array(self.norm_stats["feature_std"])
        return (features - mean) / std
    
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> tuple:
        """
        Run GNN prediction.
        
        Returns: (pm25_predictions, uncertainties)
        """
        # Normalize features
        features_norm = self.normalize_features(features)
        
        # Convert to tensor
        x = torch.tensor(features_norm, dtype=torch.float32)
        
        # Run model
        pm25_pred, uncertainty = self.model(x, self.edge_index)
        
        # Convert to numpy
        pm25_pred = pm25_pred.numpy().flatten()
        uncertainty = uncertainty.numpy().flatten()
        
        # Ensure predictions are reasonable
        pm25_pred = np.clip(pm25_pred, 0, 200)
        
        return pm25_pred, uncertainty
    
    async def update_all_predictions(self):
        """Update all predictions (called every 6 hours)"""
        try:
            logger.info("Starting full prediction update...")
            
            # Fetch weather forecasts
            logger.info("Fetching weather forecasts...")
            self.weather_cache = await fetch_weather_forecast_batch()
            logger.info(f"Weather cached for {len(self.weather_cache)} cities")
            
            # Fetch current data from Supabase
            city_data = self.fetch_current_data()
            
            # Build current predictions (hour 0)
            logger.info("Building current predictions...")
            current_results = []
            
            # Get features for current time (hour 0)
            features_now = self.build_feature_matrix(city_data, hour_offset=0)
            pm25_now, unc_now = self.predict(features_now)
            
            for i, city in enumerate(self.cities):
                if city not in city_data:
                    continue
                
                # Use GNN prediction
                predicted_pm25 = float(pm25_now[i])
                uncertainty = float(unc_now[i])
                predicted_aqi = pm25_to_aqi(predicted_pm25)
                status = aqi_to_category(predicted_aqi)
                
                weather_now = self.weather_cache.get(city, [{}])[0]
                coords = CITY_COORDINATES.get(city, {})
                
                current_results.append({
                    "city": city,
                    "pm25": round(predicted_pm25, 2),
                    "aqi": round(predicted_aqi, 1),
                    "uncertainty": round(uncertainty, 2),
                    "status": status,
                    "category": status,
                    "temperature": round(weather_now.get("temperature", 26.0), 1),
                    "latitude": coords.get("lat"),
                    "longitude": coords.get("lon"),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"  {city:15s}: PM2.5={predicted_pm25:6.2f}, AQI={predicted_aqi:5.1f}, Status={status}")
            
            self.current_predictions = current_results
            logger.info("Current predictions complete")
            
            # Build forecasts for each city
            logger.info("Building forecasts...")
            self.city_forecasts = {}
            
            # First, get CURRENT predictions to use as baseline
            features_now = self.build_feature_matrix(city_data, hour_offset=0)
            pm25_current, _ = self.predict(features_now)
            
            for city in self.cities:
                if city not in city_data:
                    continue
                
                forecast = []
                city_idx = self.city_to_idx[city]
                
                # CRITICAL FIX: Use CURRENT PREDICTION as baseline, not stale Supabase data
                current_pm25 = float(pm25_current[city_idx])
                current_pm25 = max(5.0, min(current_pm25, 150.0))  # Clamp to reasonable range
                
                logger.info(f"Forecasting {city}: baseline PM2.5 = {current_pm25:.2f}")
                
                for hour in FORECAST_HOURS:
                    # Get features for this future hour (always using current PM2.5 as baseline)
                    features = self.build_feature_matrix(city_data, hour_offset=hour)
                    pm25_pred_raw, unc_pred = self.predict(features)
                    
                    pm25_raw = float(pm25_pred_raw[city_idx])
                    
                    # ENSEMBLE: Blend GNN prediction with persistence
                    # Weight decreases with forecast horizon
                    persistence_weight = max(0.3, 1.0 - (hour / 72.0) * 0.7)  # 100% at h=0 to 30% at h=72
                    gnn_weight = 1.0 - persistence_weight
                    
                    pm25_ensemble = (persistence_weight * current_pm25) + (gnn_weight * pm25_raw)
                    
                    # Clamp predictions to reasonable range
                    pm25 = max(5.0, min(pm25_ensemble, 150.0))
                    
                    uncertainty = float(unc_pred[city_idx])
                    aqi = pm25_to_aqi(pm25)
                    
                    weather_list = self.weather_cache.get(city, [])
                    weather = weather_list[hour] if hour < len(weather_list) else {}
                    
                    forecast.append({
                        "city": city,
                        "hour": hour,
                        "pm25": round(pm25, 2),
                        "aqi": round(aqi, 1),
                        "category": aqi_to_category(aqi),
                        "temperature": round(weather.get("temperature", 26.0), 1),
                        "uncertainty": round(uncertainty, 2),
                        "timestamp": (datetime.now() + timedelta(hours=hour)).isoformat()
                    })
                
                self.city_forecasts[city] = forecast
            
            logger.info(f"Full update complete! Cached {len(self.city_forecasts)} city forecasts")
            
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=True)


# Global model manager
model_mgr = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Starting up...")
    
    # Load graph and model
    model_mgr.load_graph()
    model_mgr.load_model()
    logger.info("Loading normalization statistics...")
    
    # Initial prediction update
    await model_mgr.update_all_predictions()
    
    # Start scheduler (update every 6 hours)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        model_mgr.update_all_predictions,
        'interval',
        hours=6,
        id='update_predictions'
    )
    scheduler.start()
    logger.info("Starting on port 8080")
    logger.info("Startup complete")
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    logger.info("Shutdown complete")


# FastAPI app
app = FastAPI(
    title="HazeRadar GNN Inference API",
    description="Real-time PM2.5 forecasting for West Java cities",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://haze-radar-team-23.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class CityPrediction(BaseModel):
    city: str
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    temperature: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: str
    category: str = None


class ForecastPoint(BaseModel):
    city: str
    hour: int
    pm25: float
    aqi: float
    category: str
    temperature: float
    uncertainty: float
    timestamp: str


# API endpoints
@app.get("/")
async def root():
    return {
        "service": "HazeRadar GNN Inference API",
        "version": "2.0",
        "status": "running",
        "cities": len(model_mgr.cities) if model_mgr.cities else 0,
        "model_loaded": model_mgr.model is not None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_mgr.model is not None,
        "graph_loaded": model_mgr.graph is not None,
        "cities": len(model_mgr.cities) if model_mgr.cities else 0,
        "cached_predictions": len(model_mgr.current_predictions),
        "cached_forecasts": len(model_mgr.city_forecasts)
    }


@app.get("/cities")
async def get_cities():
    return {"cities": model_mgr.cities or []}


@app.get("/api/predictions/current", response_model=List[CityPrediction])
async def get_current_predictions():
    """Get current PM2.5 predictions for all cities"""
    if not model_mgr.current_predictions:
        raise HTTPException(status_code=503, detail="Predictions not yet available")
    
    return model_mgr.current_predictions


@app.get("/api/forecast/{city}", response_model=List[ForecastPoint])
async def get_city_forecast(city: str):
    """Get 72-hour forecast for a specific city"""
    if city not in model_mgr.city_forecasts:
        raise HTTPException(status_code=404, detail=f"No forecast for city: {city}")
    
    return model_mgr.city_forecasts[city]


@app.post("/api/refresh")
async def refresh_predictions():
    """Manually trigger prediction refresh"""
    await model_mgr.update_all_predictions()
    return {"status": "refreshed", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting on PORT: {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
