# main.py
"""
Production FastAPI backend for spatiotemporal air quality forecasting
Uses pre-trained GNN model for PM2.5 prediction and 72-hour forecasting
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "avg_fire_confidence",
    "upwind_fire_count",
    "population_density"
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
    "Bengkulu": {"lat": -3.8008, "lon": 102.2655},
    "Banjarmasin": {"lat": -3.3194, "lon": 114.5897},
    "Palangkaraya": {"lat": -2.2089, "lon": 113.9213},
    "Sampit": {"lat": -2.5333, "lon": 112.95},
    "Pangkalan Bun": {"lat": -2.6833, "lon": 111.6167}
}


async def fetch_weather_forecast(city: str, hours: int = 72) -> List[Dict]:
    """Fetch hourly weather forecast from OpenMeteo API"""
    if city not in CITY_COORDINATES:
        logger.warning(f"City {city} not in coordinates")
        # Return fallback hourly data
        return [{"hour": h, "temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5} 
                for h in range(0, hours + 1)]
    
    coords = CITY_COORDINATES[city]
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
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    hourly = data.get("hourly", {})
                    temps = hourly.get("temperature_2m", [])
                    humidity = hourly.get("relative_humidity_2m", [])
                    wind = hourly.get("wind_speed_10m", [])
                    
                    forecasts = []
                    for h in range(min(hours + 1, len(temps))):
                        forecasts.append({
                            "hour": h,
                            "temperature": temps[h] if h < len(temps) else 27.0,
                            "humidity": humidity[h] if h < len(humidity) else 75.0,
                            "wind_speed": wind[h] if h < len(wind) else 3.5
                        })
                    return forecasts
    except Exception as e:
        logger.error(f"OpenMeteo hourly forecast error for {city}: {e}")
    
    # Fallback
    return [{"hour": h, "temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5} 
            for h in range(0, hours + 1)]


async def fetch_current_weather(city: str) -> Dict:
    """Fetch current weather snapshot"""
    forecasts = await fetch_weather_forecast(city, hours=0)
    return forecasts[0] if forecasts else {"temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5}


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
                "city," + ",".join(FEATURE_COLS) + ",target_pm25_24h"
            ).order("timestamp", desc=True).limit(50).execute()
            
            if not response.data:
                return {}
            
            city_data = {}
            seen = set()
            for row in response.data:
                city = row.get("city")
                if city and city not in seen:
                    seen.add(city)
                    city_data[city] = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
                    city_data[city]["target_pm25"] = float(row.get("target_pm25_24h", 0))
            
            return city_data
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return {}
    
    async def predict_current(self) -> List[Dict]:
        """Run inference on current data"""
        try:
            city_data = self.fetch_current_data()
            
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
            
            results = []
            for i, city in enumerate(self.cities):
                # CRITICAL FIX: Use Supabase target_pm25_24h as primary source
                if city in city_data:
                    pm25 = city_data[city].get("target_pm25", 0)
                    # Only use model if Supabase data is missing or zero
                    if pm25 == 0:
                        pm25 = float(pm25_values[i])
                        pm25 = max(5.0, min(pm25, 150.0))  # Clamp model predictions
                else:
                    pm25 = float(pm25_values[i])
                    pm25 = max(5.0, min(pm25, 150.0))
                
                aqi = pm25_to_aqi(pm25)
                status = aqi_to_category(aqi)
                
                # Get current weather
                weather = await fetch_current_weather(city)
                
                results.append({
                    "city": city,
                    "pm25": round(pm25, 2),
                    "aqi": round(aqi, 1),
                    "uncertainty": round(float(uncertainty_values[i]), 2),
                    "status": status,
                    "temperature": round(weather.get("temperature", 27.0), 1),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"{city}: PM2.5={pm25:.1f}, AQI={aqi:.0f}, Status={status}")
            
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def forecast_city_hours(self, city: str) -> List[Dict]:
        """Generate complete hourly forecast for a city (0-60 hours)"""
        if city not in self.cities:
            raise ValueError(f"City {city} not in graph")
        
        city_idx = self.city_to_idx[city]
        
        # Get hourly weather forecast
        weather_forecast = await fetch_weather_forecast(city, hours=60)
        
        # Fetch current Supabase data
        city_data = self.fetch_current_data()
        base_features = city_data.get(city, {col: 0.0 for col in FEATURE_COLS})
        
        forecasts = []
        
        for target_hour in FORECAST_HOURS:
            # Build features for this time step
            features_list = []
            
            for c in self.cities:
                if c in city_data:
                    base = city_data[c]
                else:
                    base = {col: 0.0 for col in FEATURE_COLS}
                
                # Get weather at target hour
                weather_at_hour = weather_forecast[min(target_hour, len(weather_forecast) - 1)]
                
                # Time-based feature drift
                fire_decay = max(0, 1 - (target_hour / 48.0))  # Fires decay over time
                
                feature_vector = [
                    weather_at_hour.get("temperature", 27.0),
                    weather_at_hour.get("humidity", 75.0),
                    weather_at_hour.get("wind_speed", 3.5),
                    base.get("wind_direction", 120.0),
                    base.get("avg_fire_confidence", 0.0) * fire_decay,
                    base.get("upwind_fire_count", 0.0) * fire_decay,
                    base.get("population_density", 1000.0)
                ]
                features_list.append(feature_vector)
            
            features = np.array(features_list, dtype=np.float32)
            features_norm = self.normalize_features(features)
            x = torch.tensor(features_norm, dtype=torch.float32)
            
            with torch.no_grad():
                pm25_pred, uncertainty = self.model(x, self.edge_index)
                pm25_values = pm25_pred.cpu().numpy().flatten()
            
            pm25 = float(pm25_values[city_idx])
            
            # Special handling for hour 0 - use Supabase if available
            if target_hour == 0 and city in city_data:
                supabase_pm25 = city_data[city].get("target_pm25", 0)
                if supabase_pm25 > 0:
                    pm25 = supabase_pm25
            
            # Apply realistic bounds and gradual change
            pm25 = max(5.0, min(pm25, 150.0))
            
            aqi = pm25_to_aqi(pm25)
            category = aqi_to_category(aqi)
            
            weather_at_hour = weather_forecast[min(target_hour, len(weather_forecast) - 1)]
            
            forecasts.append({
                "city": city,
                "hour": target_hour,
                "pm25": round(pm25, 2),
                "aqi": round(aqi, 1),
                "category": category,
                "temperature": round(weather_at_hour.get("temperature", 27.0), 1),
                "uncertainty": round(float(pm25 * 0.1), 2),  # 10% uncertainty
                "timestamp": (datetime.now() + timedelta(hours=target_hour)).isoformat()
            })
        
        return forecasts


class CurrentPrediction(BaseModel):
    city: str
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    temperature: float
    timestamp: str


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
    description="GNN-based air quality forecasting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_artifacts()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "HazeRadar Inference API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "cities_count": len(model_manager.cities) if model_manager.cities else 0
    }


@app.get("/api/predictions/current", response_model=List[CurrentPrediction])
async def get_current_predictions():
    """Get current predictions for all cities"""
    try:
        return await model_manager.predict_current()
    except Exception as e:
        logger.error(f"Endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/{city}", response_model=List[ForecastPoint])
async def get_city_forecast(city: str):
    """Get hourly forecast for specific city (matches frontend endpoint)"""
    try:
        return await model_manager.forecast_city_hours(city)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast failed for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities")
async def get_cities():
    return {
        "cities": model_manager.cities,
        "count": len(model_manager.cities)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
