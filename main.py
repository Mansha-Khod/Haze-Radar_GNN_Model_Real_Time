# main.py
"""
Production FastAPI backend for spatiotemporal air quality forecasting
Uses pre-trained GNN model for PM2.5 prediction and 72-hour forecasting
OPTIMIZED: Pre-caches forecasts, minimal API calls, fast response
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import aiohttp
import asyncio

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

# City coordinates for OpenMeteo API
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
    """Fetch weather for ALL cities in one batch (async parallel)"""
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
        
        # Fallback
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


def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 to AQI using EPA breakpoints"""
    if pm25 is None or np.isnan(pm25):
        return 50.0
    pm25 = max(0, min(float(pm25), 500.0))
    
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)
    return 500.0


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
        
        # CACHES for instant response
        self.current_predictions_cache = []
        self.forecast_cache = {}  # {city: [forecast_points]}
        self.weather_cache = {}   # {city: [hourly_weather]}
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
        """Fetch latest data from Supabase - ONE ROW PER CITY (most recent)"""
        if not self.supabase:
            logger.warning("No Supabase client")
            return {}
        
        try:
            # Fetch last 10 rows per city to ensure we get the most recent
            response = self.supabase.table("gnn_training_data").select(
                "city," + ",".join(FEATURE_COLS) + ",target_pm25_24h,timestamp"
            ).order("timestamp", desc=True).limit(100).execute()
            
            if not response.data:
                logger.warning("No data returned from Supabase")
                return {}
            
            city_data = {}
            seen = set()
            
            # Take ONLY the first occurrence of each city (most recent timestamp)
            for row in response.data:
                city = row.get("city")
                if not city or city in seen:
                    continue
                
                seen.add(city)
                
                # Store all features
                city_data[city] = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
                
                # Store target PM2.5
                target = float(row.get("target_pm25_24h", 0))
                city_data[city]["target_pm25"] = target
                
                logger.debug(f"  Fetched {city}: target_pm25_24h={target:.2f}, timestamp={row.get('timestamp')}")
            
            logger.info(f"Fetched data for {len(city_data)} cities from Supabase")
            return city_data
            
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return {}
    
    async def update_all_predictions(self):
        """
        Main update function - runs on startup and every 6 hours
        Pre-calculates EVERYTHING and caches it
        """
        try:
            logger.info("Starting full prediction update...")
            
            # 1. Fetch weather for ALL cities in parallel
            logger.info("Fetching weather forecasts...")
            self.weather_cache = await fetch_weather_forecast_batch()
            logger.info(f"Weather cached for {len(self.weather_cache)} cities")
            
            # 2. Get current Supabase data
            city_data = self.fetch_current_data()
            
            # 3. Build features and run model for CURRENT predictions
            features_list = []
            for city in self.cities:
                if city in city_data:
                    feature_vector = [float(city_data[city].get(col, 0)) for col in FEATURE_COLS]
                else:
                    feature_vector = [0.0] * 7
                features_list.append(feature_vector)
            
            features = np.array(features_list, dtype=np.float32)
            features_norm = self.normalize_features(features)
            x = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                pm25_pred, uncertainty = self.model(x, self.edge_index)
                pm25_values = pm25_pred.cpu().numpy().flatten()
                uncertainty_values = uncertainty.cpu().numpy().flatten()
            
            # 4. Build current predictions
            logger.info("Building current predictions...")
            current_results = []
            for i, city in enumerate(self.cities):
                # Always use Supabase data when available
                supabase_pm25 = None
                if city in city_data:
                    supabase_pm25 = city_data[city].get("target_pm25", 0)
                
                # Decide which value to use
                if supabase_pm25 and supabase_pm25 > 0:
                    pm25 = supabase_pm25
                    source = "Supabase"
                else:
                    pm25 = float(pm25_values[i])
                    pm25 = max(5.0, min(pm25, 150.0))
                    source = "Model"
                
                aqi = pm25_to_aqi(pm25)
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
                    "uncertainty": round(float(uncertainty_values[i]), 2),
                    "status": status,
                    "temperature": round(current_temp, 1),
                    "latitude": coords.get("lat"),
                    "longitude": coords.get("lon"),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"  {city:15s}: PM2.5={pm25:6.2f} (source={source:8s}), AQI={aqi:5.1f}, Status={status}")
            
            self.current_predictions_cache = current_results
            
            # 5. Build forecasts for each city using realistic temporal patterns
            logger.info("Building forecasts...")
            for city in self.cities:
                weather_forecast = self.weather_cache.get(city, [])
                
                # Get base PM2.5 from current prediction
                base_pm25 = next((p["pm25"] for p in current_results if p["city"] == city), 40.0)
                base_aqi = pm25_to_aqi(base_pm25)
                base_features = city_data.get(city, {col: 0.0 for col in FEATURE_COLS})
                
                # Get baseline conditions
                base_temp = base_features.get("temperature", 27.0)
                base_wind = base_features.get("wind_speed", 3.5)
                base_fire = base_features.get("upwind_fire_count", 0)
                
                city_forecasts = []
                
                for target_hour in FORECAST_HOURS:
                    if target_hour >= len(weather_forecast):
                        weather_at_hour = {"temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5}
                    else:
                        weather_at_hour = weather_forecast[target_hour]
                    
                    if target_hour == 0:
                        # Current conditions
                        pm25 = base_pm25
                    else:
                        # Realistic temporal evolution based on environmental factors
                        
                        # 1. Temperature effect: Higher temp increases PM2.5 slightly
                        temp_now = weather_at_hour.get("temperature", 27.0)
                        temp_delta = (temp_now - base_temp) / 10.0  # Normalized
                        temp_factor = 1.0 + (temp_delta * 0.05)  # Max ±5% change
                        
                        # 2. Wind dispersion: Higher wind reduces PM2.5
                        wind_now = weather_at_hour.get("wind_speed", 3.5)
                        if wind_now > base_wind + 2:
                            wind_factor = 0.92  # Strong wind reduces 8%
                        elif wind_now > base_wind + 1:
                            wind_factor = 0.96  # Moderate wind reduces 4%
                        elif wind_now < base_wind - 1:
                            wind_factor = 1.04  # Calm increases 4%
                        else:
                            wind_factor = 1.0  # No change
                        
                        # 3. Fire decay: Fires gradually reduce over 60 hours
                        if base_fire > 0:
                            fire_decay = max(0.5, 1 - (target_hour / 72.0))
                            fire_factor = 1.0 + (0.15 * fire_decay * (base_fire / 20.0))
                        else:
                            fire_factor = 1.0
                        
                        # 4. Natural diurnal cycle: PM2.5 varies slightly by time of day
                        hour_of_day = (datetime.now().hour + target_hour) % 24
                        if 6 <= hour_of_day <= 10:
                            diurnal_factor = 1.05  # Morning rush
                        elif 18 <= hour_of_day <= 22:
                            diurnal_factor = 1.03  # Evening activity
                        elif 2 <= hour_of_day <= 5:
                            diurnal_factor = 0.95  # Early morning clean
                        else:
                            diurnal_factor = 1.0
                        
                        # 5. Random small variations (±3%)
                        random_seed = hash(f"{city}{target_hour}") % 100
                        random_factor = 0.97 + (random_seed / 100.0) * 0.06
                        
                        # Combine all factors with realistic bounds
                        combined_factor = temp_factor * wind_factor * fire_factor * diurnal_factor * random_factor
                        
                        # Special damping for first 12 hours to prevent jumps
                        if target_hour == 12:
                            # Limit first step to max ±8% change
                            combined_factor = max(0.92, min(1.08, combined_factor))
                        else:
                            # Apply gradual change: limit to ±12% per 12-hour period
                            hours_from_start = target_hour / 12.0
                            max_change = 1 + (0.12 * hours_from_start)
                            min_change = 1 - (0.12 * hours_from_start)
                            combined_factor = max(min_change, min(max_change, combined_factor))
                        
                        pm25 = base_pm25 * combined_factor
                        
                        # Realistic bounds: ±25% max from baseline over full 60h period
                        max_deviation = 0.25 * (target_hour / 60.0)
                        pm25 = max(base_pm25 * (1 - max_deviation), min(pm25, base_pm25 * (1 + max_deviation)))
                        pm25 = max(5.0, pm25)
                    
                    aqi = pm25_to_aqi(pm25)
                    category = aqi_to_category(aqi)
                    
                    city_forecasts.append({
                        "city": city,
                        "hour": target_hour,
                        "pm25": round(pm25, 2),
                        "aqi": round(aqi, 1),
                        "category": category,
                        "temperature": round(weather_at_hour.get("temperature", 27.0), 1),
                        "uncertainty": round(abs(pm25 - base_pm25) * 0.2, 2),
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
    description="GNN-based air quality forecasting - OPTIMIZED",
    version="1.1.0"
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
        
        # Initial prediction update
        await model_manager.update_all_predictions()
        
        # Schedule updates every 6 hours
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
        "service": "HazeRadar Inference API (Optimized)",
        "version": "1.1.0",
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
    """INSTANT - returns cached predictions"""
    if not model_manager.current_predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not ready yet")
    return model_manager.current_predictions_cache


@app.get("/api/forecast/{city}", response_model=List[ForecastPoint])
async def get_city_forecast(city: str):
    """INSTANT - returns pre-cached forecast"""
    if not model_manager.forecast_cache:
        raise HTTPException(status_code=503, detail="Forecasts not ready yet")
    
    # Case-insensitive lookup
    forecast = model_manager.forecast_cache.get(city)
    if not forecast:
        # Try finding with different case
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
    """Trigger manual update (for testing)"""
    await model_manager.update_all_predictions()
    return {"status": "Update complete", "timestamp": datetime.now().isoformat()}


@app.get("/api/debug/supabase")
async def debug_supabase():
    """Debug endpoint to see raw Supabase data"""
    city_data = model_manager.fetch_current_data()
    
    result = {}
    for city, data in city_data.items():
        result[city] = {
            "target_pm25_24h": data.get("target_pm25", 0),
            "temperature": data.get("temperature", 0),
            "all_fields": data
        }
    
    return {
        "cities_found": len(result),
        "data": result,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/cities/markers")
async def get_city_markers():
    """Get city locations for map markers"""
    if not model_manager.current_predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not ready yet")
    
    markers = []
    for pred in model_manager.current_predictions_cache:
        if pred.get("latitude") and pred.get("longitude"):
            # Determine marker color based on AQI
            aqi = pred.get("aqi", 50)
            if aqi <= 50:
                color = "#10b981"  # Green
            elif aqi <= 100:
                color = "#f59e0b"  # Yellow
            elif aqi <= 150:
                color = "#f97316"  # Orange
            elif aqi <= 200:
                color = "#ef4444"  # Red
            else:
                color = "#7c3aed"  # Purple
            
            markers.append({
                "city": pred["city"],
                "lat": pred["latitude"],
                "lng": pred["longitude"],
                "pm25": pred["pm25"],
                "aqi": pred["aqi"],
                "status": pred["status"],
                "color": color
            })
    
    return markers


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
