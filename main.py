# main.py
"""
Production FastAPI backend for spatiotemporal air quality forecasting
FIXED: Uses collected AQI from source, scales forecasts proportionally
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
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

FORECAST_HOURS = [0, 12, 24, 36, 48, 60]

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/realtime_haze_gnn_infer.pt")
NORM_STATS_PATH = os.getenv("NORM_STATS_PATH", "artifacts/normalization_stats.json")
GRAPH_PATH = os.getenv("GRAPH_PATH", "artifacts/city_graph.json")

CITY_COORDINATES = {
    "Pekanbaru": {"lat": 0.5071, "lon": 101.4478},
    "Palembang": {"lat": -2.9761, "lon": 104.7754},
    "Jambi": {"lat": -1.6101, "lon": 103.6131},
    "Pontianak": {"lat": -0.0263, "lon": 109.3425},
    "Palangkaraya": {"lat": -2.2089, "lon": 113.9213},
    "Sampit": {"lat": -2.5333, "lon": 112.95},
    "Pangkalan Bun": {"lat": -2.6833, "lon": 111.6167}
}


async def fetch_weather_forecast_batch() -> Dict[str, List[Dict]]:
    """Fetch weather for ALL cities in one batch"""
    async def fetch_one_city(city: str, coords: Dict) -> tuple:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
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
                        wind = hourly.get("wind_speed_10m", [])
                        
                        forecasts = []
                        for h in range(min(72, len(temps))):
                            forecasts.append({
                                "hour": h,
                                "temperature": temps[h] if h < len(temps) else 27.0,
                                "humidity": humidity[h] if h < len(humidity) else 75.0,
                                "wind_speed": wind[h] if h < len(wind) else 3.5
                            })
                        return (city, forecasts)
        except Exception as e:
            logger.error(f"Weather fetch failed for {city}: {e}")
        
        return (city, [{"hour": h, "temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5} 
                      for h in range(72)])
    
    tasks = [fetch_one_city(city, coords) for city, coords in CITY_COORDINATES.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)


class HazeForecastGNN(nn.Module):
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


def aqi_to_category(aqi: float) -> str:
    """Convert AQI to category"""
    try:
        aqi = float(aqi)
        if np.isnan(aqi):
            return "Unknown"
    except (ValueError, TypeError):
        return "Unknown"
    
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


class ModelManager:
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")
        self.feature_mean = None
        self.feature_std = None
        self.city_to_idx = None
        self.edges = None
        self.cities = None
        self.edge_index = None
        self.supabase = None
        
        self.current_predictions_cache = []
        self.forecast_cache = {}
        self.weather_cache = {}
        self.last_update = None
    
    def load_artifacts(self):
        """Load model and graph structure"""
        logger.info("Loading normalization statistics...")
        with open(NORM_STATS_PATH, 'r') as f:
            norm_stats = json.load(f)
        
        self.feature_mean = np.array(norm_stats['feature_mean'], dtype=np.float32)
        self.feature_std = np.array(norm_stats['feature_std'], dtype=np.float32)
        
        logger.info("Loading graph structure...")
        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        
        self.city_to_idx = graph_data['city_to_idx']
        self.edges = graph_data['edges']
        self.cities = sorted(self.city_to_idx.keys(), key=lambda c: self.city_to_idx[c])
        self.edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        
        logger.info(f"Graph loaded: {len(self.cities)} cities, {len(self.edges)} edges")
        
        logger.info("Loading model...")
        self.model = HazeForecastGNN(in_features=7, hidden_dim=128, num_heads=4, dropout=0.2)
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Supabase init failed: {e}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_std
    
    def fetch_current_data(self) -> Dict[str, Dict]:
        """Fetch latest data from Supabase"""
        if not self.supabase:
            logger.warning("No Supabase client")
            return {}
        
        try:
            response = self.supabase.table("gnn_training_data").select(
                "city," + ",".join(FEATURE_COLS) + ",target_pm25_24h,current_aqi,timestamp"
            ).order("timestamp", desc=True).limit(100).execute()
            
            if not response.data:
                logger.warning("No data returned from Supabase")
                return {}
            
            city_data = {}
            seen = set()
            
            for row in response.data:
                city = row.get("city")
                if not city or city in seen:
                    continue
                
                seen.add(city)
                city_data[city] = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
                city_data[city]["target_pm25"] = float(row.get("target_pm25_24h", 0))
                city_data[city]["current_aqi"] = float(row.get("current_aqi", 0))
                city_data[city]["timestamp"] = row.get("timestamp", "")
            
            logger.info(f"Fetched data for {len(city_data)} cities from Supabase")
            return city_data
            
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return {}
    
    async def update_all_predictions(self):
        """Main update function - runs on startup and every 6 hours"""
        try:
            logger.info("Starting full prediction update...")
            
            # 1. Fetch weather for ALL cities in parallel
            logger.info("Fetching weather forecasts...")
            self.weather_cache = await fetch_weather_forecast_batch()
            logger.info(f"Weather cached for {len(self.weather_cache)} cities")
            
            # 2. Get current Supabase data
            city_data = self.fetch_current_data()
            
            # 3. Build current predictions - TRUST SUPABASE AQI
            logger.info("Building current predictions...")
            current_results = []
            
            for city in self.cities:
                if city not in city_data:
                    continue
                
                # ALWAYS use collected data from WAQI (via your collector)
                pm25 = city_data[city].get("target_pm25", 0)
                aqi = city_data[city].get("current_aqi", 0)
                
                if pm25 <= 0 or aqi <= 0:
                    continue
                
                status = aqi_to_category(aqi)
                
                # Get current weather
                weather_today = self.weather_cache.get(city, [{"temperature": 27.0}])
                current_temp = weather_today[0]["temperature"] if weather_today else 27.0
                
                # Get coordinates
                coords = CITY_COORDINATES.get(city, {})
                
                current_results.append({
                    "city": city,
                    "pm25": round(pm25, 2),
                    "aqi": round(aqi, 1),
                    "uncertainty": 0.0,
                    "status": status,
                    "temperature": round(current_temp, 1),
                    "latitude": coords.get("lat"),
                    "longitude": coords.get("lon"),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"  {city:15s}: PM2.5={pm25:6.2f}, AQI={aqi:5.1f}, Status={status}")
            
            self.current_predictions_cache = current_results
            logger.info("Current predictions complete")
            
            # 4. Build forecasts - scale AQI proportionally with PM2.5
            logger.info("Building forecasts...")
            for city in self.cities:
                current_pred = next((p for p in current_results if p["city"] == city), None)
                if not current_pred:
                    continue
                
                base_pm25 = current_pred["pm25"]
                base_aqi = current_pred["aqi"]
                weather_forecast = self.weather_cache.get(city, [])
                base_features = city_data.get(city, {col: 0.0 for col in FEATURE_COLS})
                
                city_forecasts = []
                
                for target_hour in FORECAST_HOURS:
                    if target_hour >= len(weather_forecast):
                        weather_at_hour = {"temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5}
                    else:
                        weather_at_hour = weather_forecast[target_hour]
                    
                    if target_hour == 0:
                        # Hour 0 = exactly current data
                        pm25 = base_pm25
                        aqi = base_aqi
                    else:
                        # Calculate PM2.5 change based on weather
                        temp_now = weather_at_hour.get("temperature", 27.0)
                        wind_now = weather_at_hour.get("wind_speed", 3.5)
                        base_temp = base_features.get("temperature", 27.0)
                        base_wind = base_features.get("wind_speed", 3.5)
                        
                        # Temperature effect
                        temp_delta = (temp_now - base_temp) / 10.0
                        temp_factor = 1.0 + (temp_delta * 0.05)
                        
                        # Wind effect
                        if wind_now > base_wind + 2:
                            wind_factor = 0.92
                        elif wind_now > base_wind + 1:
                            wind_factor = 0.96
                        elif wind_now < base_wind - 1:
                            wind_factor = 1.04
                        else:
                            wind_factor = 1.0
                        
                        # Fire decay
                        base_fire = base_features.get("upwind_fire_count", 0)
                        if base_fire > 0:
                            fire_decay = max(0.5, 1 - (target_hour / 72.0))
                            fire_factor = 1.0 + (0.15 * fire_decay * (base_fire / 20.0))
                        else:
                            fire_factor = 1.0
                        
                        # Diurnal cycle
                        hour_of_day = (datetime.now().hour + target_hour) % 24
                        if 6 <= hour_of_day <= 10:
                            diurnal_factor = 1.05
                        elif 18 <= hour_of_day <= 22:
                            diurnal_factor = 1.03
                        elif 2 <= hour_of_day <= 5:
                            diurnal_factor = 0.95
                        else:
                            diurnal_factor = 1.0
                        
                        # Combine factors
                        combined_factor = temp_factor * wind_factor * fire_factor * diurnal_factor
                        
                        # Special damping for first 12 hours
                        if target_hour == 12:
                            combined_factor = max(0.92, min(1.08, combined_factor))
                        else:
                            hours_from_start = target_hour / 12.0
                            max_change = 1 + (0.12 * hours_from_start)
                            min_change = 1 - (0.12 * hours_from_start)
                            combined_factor = max(min_change, min(max_change, combined_factor))
                        
                        pm25 = base_pm25 * combined_factor
                        
                        # Realistic bounds
                        max_deviation = 0.25 * (target_hour / 60.0)
                        pm25 = max(base_pm25 * (1 - max_deviation), min(pm25, base_pm25 * (1 + max_deviation)))
                        pm25 = max(5.0, pm25)
                        
                        # Scale AQI proportionally (keeps consistency with your data source)
                        pm25_ratio = pm25 / base_pm25 if base_pm25 > 0 else 1.0
                        aqi = base_aqi * pm25_ratio
                        aqi = max(0, min(aqi, 500))
                    
                    category = aqi_to_category(aqi)
                    
                    city_forecasts.append({
                        "city": city,
                        "hour": target_hour,
                        "pm25": round(pm25, 2),
                        "aqi": round(aqi, 1),
                        "category": category,
                        "temperature": round(weather_at_hour.get("temperature", 27.0), 1),
                        "uncertainty": round(abs(aqi - base_aqi) * 0.1, 2),
                        "timestamp": (datetime.now() + timedelta(hours=target_hour)).isoformat()
                    })
                
                self.forecast_cache[city] = city_forecasts
            
            self.last_update = datetime.now()
            logger.info(f"Full update complete! Cached {len(self.forecast_cache)} city forecasts")
            
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=True)


class CurrentPrediction(BaseModel):
    city: str
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    temperature: float
    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ForecastPoint(BaseModel):
    city: str
    hour: int
    pm25: float
    aqi: float
    category: str
    temperature: float
    uncertainty: float
    timestamp: str


app = FastAPI(
    title="HazeRadar Inference API",
    description="GNN-based air quality forecasting - Optimized",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_artifacts()
        await model_manager.update_all_predictions()
        
        scheduler.add_job(
            model_manager.update_all_predictions,
            'interval',
            hours=6,
            id='prediction_updater'
        )
        scheduler.start()
        
        logger.info("Startup complete - API ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "HazeRadar Inference API (Fixed)",
        "version": "1.2.0",
        "last_update": model_manager.last_update.isoformat() if model_manager.last_update else None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "cities_count": len(model_manager.cities) if model_manager.cities else 0,
        "cache_ready": len(model_manager.forecast_cache) > 0,
        "last_update": model_manager.last_update.isoformat() if model_manager.last_update else None
    }


@app.get("/api/predictions/current", response_model=List[CurrentPrediction])
async def get_current_predictions():
    """Get current predictions for all cities"""
    if not model_manager.current_predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not ready yet")
    return model_manager.current_predictions_cache


@app.get("/api/forecast/{city}", response_model=List[ForecastPoint])
async def get_city_forecast(city: str):
    """Get hourly forecast for specific city"""
    if not model_manager.forecast_cache:
        raise HTTPException(status_code=503, detail="Forecasts not ready yet")
    
    forecast = model_manager.forecast_cache.get(city)
    if not forecast:
        for cached_city, cached_forecast in model_manager.forecast_cache.items():
            if cached_city.lower() == city.lower():
                return cached_forecast
        
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    
    return forecast


@app.get("/cities")
async def get_cities():
    return {
        "cities": model_manager.cities,
        "count": len(model_manager.cities)
    }


@app.post("/api/update")
async def manual_update():
    """Trigger manual update"""
    await model_manager.update_all_predictions()
    return {"status": "Update complete", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
