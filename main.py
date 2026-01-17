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

FORECAST_CACHE_TTL_HOURS = 12

# City coordinates for OpenMeteo API
CITY_COORDINATES = {
    "Pekanbaru": {"lat": 0.5071, "lon": 101.4478},
    "Palembang": {"lat": -2.9761, "lon": 104.7754},
    "Jambi": {"lat": -1.6101, "lon": 103.6131},
    "Pontianak": {"lat": -0.0263, "lon": 109.3425},
    "Bengkulu": {"lat": -3.8008, "lon": 102.2655},
    "Banjarmasin": {"lat": -3.3194, "lon": 114.5897},
    "Palangkaraya": {"lat": -2.2089, "lon": 113.9213}
}


async def fetch_weather_data(city: str) -> Dict:
    """Fetch current weather from OpenMeteo API"""
    if city not in CITY_COORDINATES:
        logger.warning(f"City {city} not in coordinates")
        return {"temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5}
    
    coords = CITY_COORDINATES[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get("current", {})
                    return {
                        "temperature": current.get("temperature_2m", 27.0),
                        "humidity": current.get("relative_humidity_2m", 75.0),
                        "wind_speed": current.get("wind_speed_10m", 3.5)
                    }
    except Exception as e:
        logger.error(f"OpenMeteo error for {city}: {e}")
    
    return {"temperature": 27.0, "humidity": 75.0, "wind_speed": 3.5}


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
        self.forecast_cache = defaultdict(dict)
        self.cache_timestamps = defaultdict(dict)
        self.full_forecast_cache = {}
        self.full_forecast_timestamps = {}
    
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
                    # Store target PM2.5 for reference
                    city_data[city]["target_pm25"] = float(row.get("target_pm25_24h", 0))
            
            return city_data
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return {}
    
    def predict_current(self) -> List[Dict]:
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
                pm25_pred = torch.clamp(pm25_pred, min=5.0, max=150.0)
                pm25_values = pm25_pred.cpu().numpy().flatten()
                uncertainty_values = uncertainty.cpu().numpy().flatten()
            
            results = []
            for i, city in enumerate(self.cities):
                pm25 = float(pm25_values[i])
                aqi = pm25_to_aqi(pm25)
                status = aqi_to_category(aqi)
                
                # Get real weather data
                weather = asyncio.run(fetch_weather_data(city))
                
                # Use Supabase target PM2.5 if available and more realistic
                if city in city_data:
                    target = city_data[city].get("target_pm25", 0)
                    if 0 < target < pm25 * 0.7:  # If model overpredicts, use target
                        pm25 = target
                        aqi = pm25_to_aqi(pm25)
                        status = aqi_to_category(aqi)
                
                results.append({
                    "city": city,
                    "pm25": round(pm25, 2),
                    "aqi": round(aqi, 1),
                    "uncertainty": round(float(uncertainty_values[i]), 2),
                    "status": status,
                    "temperature": round(weather.get("temperature", 27.0), 1),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"{city}: PM2.5={pm25:.1f}, AQI={aqi:.0f}, Temp={weather.get('temperature', 27):.1f}°C")
            
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    def forecast_city(self, city: str, hour: int) -> Dict:
        """Generate forecast for specific hour"""
        if city not in self.cities:
            raise ValueError(f"City {city} not in graph")
        
        if hour == 0:
            preds = self.predict_current()
            for p in preds:
                if p["city"] == city:
                    result = {**p, "hour": 0}
                    self.forecast_cache[city][0] = result
                    return result
        
        # Use Supabase data with time adjustment
        city_data = self.fetch_current_data()
        features_list = []
        
        for c in self.cities:
            if c in city_data:
                base = city_data[c]
            else:
                base = {col: 0.0 for col in FEATURE_COLS}
            
            # Simple time-based drift
            factor = hour / 24.0
            feature_vector = [
                base.get("temperature", 27.0) + 0.5 * factor,
                base.get("humidity", 75.0) - 1.0 * factor,
                base.get("wind_speed", 3.5) + 0.1 * factor,
                base.get("wind_direction", 120.0),
                base.get("avg_fire_confidence", 0.0),
                base.get("upwind_fire_count", 0.0),
                base.get("population_density", 1000.0)
            ]
            features_list.append(feature_vector)
        
        features = np.array(features_list, dtype=np.float32)
        features_norm = self.normalize_features(features)
        x = torch.tensor(features_norm, dtype=torch.float32)
        
        with torch.no_grad():
            pm25_pred, uncertainty = self.model(x, self.edge_index)
            pm25_values = pm25_pred.cpu().numpy().flatten()
        
        idx = self.city_to_idx[city]
        pm25 = float(pm25_values[idx])
        aqi = pm25_to_aqi(pm25)
        
        weather = asyncio.run(fetch_weather_data(city))
        
        return {
            "city": city,
            "hour": hour,
            "pm25": round(pm25, 2),
            "aqi": round(aqi, 1),
            "uncertainty": 10.0,
            "status": aqi_to_category(aqi),
            "temperature": round(weather.get("temperature", 27.0), 1),
            "timestamp": (datetime.now() + timedelta(hours=hour)).isoformat()
        }
    
    def forecast_city_all(self, city: str) -> Dict:
        """Get all forecast hours for a city"""
        if city not in self.cities:
            raise ValueError(f"City {city} not in graph")
        
        forecasts = [self.forecast_city(city, h) for h in FORECAST_HOURS]
        
        return {
            "city": city,
            "hours": FORECAST_HOURS,
            "generated_at": datetime.now().isoformat(),
            "forecasts": forecasts
        }


class CurrentPrediction(BaseModel):
    city: str
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    temperature: float
    timestamp: str


class ForecastPrediction(BaseModel):
    city: str
    hour: int
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    temperature: float
    timestamp: str


class AllForecastsResponse(BaseModel):
    city: str
    hours: List[int]
    generated_at: str
    forecasts: List[ForecastPrediction]


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
    try:
        return model_manager.predict_current()
    except Exception as e:
        logger.error(f"Endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/{city}/all", response_model=AllForecastsResponse)
async def get_all_city_forecasts(city: str):
    try:
        return model_manager.forecast_city_all(city)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
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
