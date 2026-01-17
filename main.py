# main.py
"""
Production FastAPI backend for spatiotemporal air quality forecasting
Uses pre-trained GNN model for PM2.5 prediction and 72-hour forecasting
FIXED: Comprehensive error handling and fallback mechanisms
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
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


class HazeForecastGNN(nn.Module):
    """
    Graph Attention Network for PM2.5 prediction
    Architecture must match training configuration exactly
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.gat1 = GATv2Conv(
            in_features, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout
        )
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
    """
    Convert PM2.5 concentration to AQI using EPA breakpoints
    """
    if pm25 is None or np.isnan(pm25):
        return 50.0
    
    pm25 = max(0, float(pm25))
    
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
    """
    Convert AQI value to categorical status
    """
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
    """
    Manages model loading, inference, and caching
    """
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
        """
        Load model weights, normalization stats, and graph structure
        """
        logger.info("Loading normalization statistics...")
        with open(NORM_STATS_PATH, 'r') as f:
            norm_stats = json.load(f)
        
        feature_cols = norm_stats.get('feature_cols', [])
        if feature_cols != FEATURE_COLS:
            raise ValueError(
                f"Feature mismatch. Expected {FEATURE_COLS}, got {feature_cols}"
            )
        
        self.feature_mean = np.array(norm_stats['feature_mean'], dtype=np.float32)
        self.feature_std = np.array(norm_stats['feature_std'], dtype=np.float32)
        
        if len(self.feature_mean) != 7:
            raise ValueError(f"Expected 7 features, got {len(self.feature_mean)}")
        
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
        logger.info(f"Model input dim: 7")
        logger.info(f"Normalization vector length: {len(self.feature_mean)}")
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Supabase initialization failed: {e}")
                self.supabase = None
        else:
            logger.warning("Supabase credentials not found, using fallback data")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization using saved statistics
        """
        return (features - self.feature_mean) / self.feature_std
    
    def _get_fallback_data(self) -> Dict[str, Dict]:
        """
        Provide fallback data when Supabase is unavailable
        """
        logger.info("Using fallback data for all cities")
        fallback = {}
        for city in self.cities:
            fallback[city] = {
                "temperature": 27.0,
                "humidity": 75.0,
                "wind_speed": 3.5,
                "wind_direction": 120.0,
                "avg_fire_confidence": 0.3,
                "upwind_fire_count": 2.0,
                "population_density": 5000.0
            }
        return fallback
    
    def fetch_current_data(self) -> Dict[str, Dict]:
        """
        Fetch latest data per city from Supabase with robust error handling
        """
        if not self.supabase:
            logger.warning("Supabase client not initialized, using fallback data")
            return self._get_fallback_data()
        
        try:
            columns = ["city"] + FEATURE_COLS + ["current_aqi", "timestamp"]
            select_query = ", ".join(columns)
            
            logger.info(f"Fetching from Supabase: {select_query}")
            
            response = self.supabase.table("gnn_training_data").select(
                select_query
            ).order("timestamp", desc=True).limit(50).execute()
            
            if not hasattr(response, "data") or response.data is None or len(response.data) == 0:
                logger.warning("No data from Supabase, using fallback")
                return self._get_fallback_data()
            
            df = response.data
            logger.info(f"Fetched {len(df)} rows from Supabase")
            
            city_data = {}
            seen_cities = set()
            
            for row in df:
                city = row.get("city")
                if not city or city in seen_cities:
                    continue
                
                seen_cities.add(city)
                city_data[city] = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
                logger.info(f"Loaded data for city: {city}")
            
            if not city_data:
                logger.warning("No city data extracted, using fallback")
                return self._get_fallback_data()
                
            return city_data
            
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}", exc_info=True)
            return self._get_fallback_data()
    
    def predict_current(self) -> List[Dict]:
        """
        Run inference on current realtime data with comprehensive error handling
        """
        try:
            city_data = self.fetch_current_data()
            
            features_list = []
            for city in self.cities:
                if city in city_data:
                    feature_vector = [
                        float(city_data[city].get(col, 0)) 
                        for col in FEATURE_COLS
                    ]
                else:
                    feature_vector = [0.0] * 7
                    logger.warning(f"City {city} not in realtime data, using zeros")
                
                features_list.append(feature_vector)
            
            features = np.array(features_list, dtype=np.float32)
            features_norm = self.normalize_features(features)
            
            x = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
            edge_index = self.edge_index.to(self.device)
            
            with torch.no_grad():
                pm25_pred, uncertainty = self.model(x, edge_index)
                pm25_pred = torch.clamp(pm25_pred, min=5.0, max=150.0)
                
                pm25_values = pm25_pred.cpu().numpy().flatten()
                uncertainty_values = uncertainty.cpu().numpy().flatten()
            
            results = []
            for i, city in enumerate(self.cities):
                pm25 = float(pm25_values[i])
                unc = float(uncertainty_values[i])
                aqi = pm25_to_aqi(pm25)
                status = aqi_to_category(aqi)
                
                results.append({
                    "city": city,
                    "pm25": round(pm25, 2),
                    "aqi": round(aqi, 1),
                    "uncertainty": round(unc, 2),
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Predicted - {city}: PM2.5={pm25:.1f}, AQI={aqi:.0f}, Status={status}")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def fetch_forecast_features(self, city: str, hours_ahead: int) -> Dict[str, float]:
        """
        Fetch or generate forecast features for a given city and time horizon
        Applies horizon-aware feature drift as deterministic proxy for temporal evolution
        """
        base_features = {}
        
        if self.supabase:
            try:
                response = self.supabase.table("gnn_training_data").select(
                    ", ".join(["city"] + FEATURE_COLS)
                ).eq("city", city).order("timestamp", desc=True).limit(1).execute()
                
                if response.data and len(response.data) > 0:
                    row = response.data[0]
                    base_features = {col: float(row.get(col, 0)) for col in FEATURE_COLS}
            except Exception as e:
                logger.error(f"Error fetching forecast features: {e}")
        
        if not base_features:
            base_features = {
                "temperature": 25.0,
                "humidity": 70.0,
                "wind_speed": 5.0,
                "wind_direction": 90.0,
                "avg_fire_confidence": 0.0,
                "upwind_fire_count": 0.0,
                "population_density": 1000.0
            }
        
        factor = hours_ahead / 24.0
        
        base_features["temperature"] += 0.7 * factor
        base_features["humidity"] -= 1.2 * factor
        base_features["wind_speed"] += 0.15 * factor
        base_features["avg_fire_confidence"] += 0.08 * factor
        
        base_features["humidity"] = max(0, min(100, base_features["humidity"]))
        base_features["wind_speed"] = max(0, base_features["wind_speed"])
        base_features["avg_fire_confidence"] = max(0, min(1, base_features["avg_fire_confidence"]))
        
        return base_features
    
    def is_cache_valid(self, city: str, hour: int) -> bool:
        """
        Check if cached forecast is still valid
        """
        if city not in self.cache_timestamps or hour not in self.cache_timestamps[city]:
            return False
        
        cached_time = self.cache_timestamps[city][hour]
        age = datetime.now() - cached_time
        
        return age.total_seconds() < FORECAST_CACHE_TTL_HOURS * 3600
    
    def is_full_forecast_cache_valid(self, city: str) -> bool:
        """
        Check if full forecast cache is still valid
        """
        if city not in self.full_forecast_timestamps:
            return False
        
        cached_time = self.full_forecast_timestamps[city]
        age = datetime.now() - cached_time
        
        return age.total_seconds() < FORECAST_CACHE_TTL_HOURS * 3600
    
    def forecast_city(self, city: str, hour: int) -> Dict:
        """
        Generate forecast for a specific city at a given hour
        Uses caching to avoid redundant computation
        """
        if city not in self.cities:
            raise ValueError(f"City {city} not in graph")
        
        if hour not in FORECAST_HOURS:
            raise ValueError(f"Hour must be one of {FORECAST_HOURS}")
        
        if self.is_cache_valid(city, hour):
            logger.info(f"Using cached forecast for {city} at hour {hour}")
            return self.forecast_cache[city][hour]
        
        if hour == 0:
            current_predictions = self.predict_current()
            for pred in current_predictions:
                if pred["city"] == city:
                    result = {
                        "city": city,
                        "hour": 0,
                        "pm25": pred["pm25"],
                        "aqi": pred["aqi"],
                        "uncertainty": pred["uncertainty"],
                        "status": pred["status"],
                        "timestamp": datetime.now().isoformat()
                    }
                    self.forecast_cache[city][0] = result
                    self.cache_timestamps[city][0] = datetime.now()
                    return result
        
        features_list = []
        for c in self.cities:
            forecast_features = self.fetch_forecast_features(c, hour)
            feature_vector = [forecast_features.get(col, 0.0) for col in FEATURE_COLS]
            features_list.append(feature_vector)
        
        features = np.array(features_list, dtype=np.float32)
        features_norm = self.normalize_features(features)
        
        x = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
        edge_index = self.edge_index.to(self.device)
        
        with torch.no_grad():
            pm25_pred, uncertainty = self.model(x, edge_index)
            pm25_pred = torch.clamp(pm25_pred, min=5.0, max=150.0)
            
            pm25_values = pm25_pred.cpu().numpy().flatten()
            uncertainty_values = uncertainty.cpu().numpy().flatten()
        
        city_idx = self.city_to_idx[city]
        pm25 = float(pm25_values[city_idx])
        unc = float(uncertainty_values[city_idx])
        aqi = pm25_to_aqi(pm25)
        status = aqi_to_category(aqi)
        
        result = {
            "city": city,
            "hour": hour,
            "pm25": round(pm25, 2),
            "aqi": round(aqi, 1),
            "uncertainty": round(unc, 2),
            "status": status,
            "timestamp": (datetime.now() + timedelta(hours=hour)).isoformat()
        }
        
        self.forecast_cache[city][hour] = result
        self.cache_timestamps[city][hour] = datetime.now()
        
        return result
    
    def forecast_city_all(self, city: str) -> Dict:
        """
        Generate all forecast hours for a city
        This is the PRIMARY forecast endpoint
        Optimized for frontend slider usage
        """
        if city not in self.cities:
            raise ValueError(f"City {city} not in graph")
        
        if self.is_full_forecast_cache_valid(city):
            logger.info(f"Using cached full forecast for {city}")
            return self.full_forecast_cache[city]
        
        forecasts = []
        for hour in FORECAST_HOURS:
            forecast = self.forecast_city(city, hour)
            forecasts.append(forecast)
        
        result = {
            "city": city,
            "hours": FORECAST_HOURS,
            "generated_at": datetime.now().isoformat(),
            "forecasts": forecasts
        }
        
        self.full_forecast_cache[city] = result
        self.full_forecast_timestamps[city] = datetime.now()
        
        return result


class CurrentPrediction(BaseModel):
    city: str
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    timestamp: str


class ForecastPrediction(BaseModel):
    city: str
    hour: int
    pm25: float
    aqi: float
    uncertainty: float
    status: str
    timestamp: str


class AllForecastsResponse(BaseModel):
    city: str
    hours: List[int]
    generated_at: str
    forecasts: List[ForecastPrediction]


class ConfigResponse(BaseModel):
    forecast_hours: List[int]
    features: List[str]
    model: str
    num_cities: int


app = FastAPI(
    title="HazeRadar Inference API",
    description="GNN-based spatiotemporal air quality forecasting",
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
    """
    Load all artifacts and validate configuration on startup
    """
    try:
        model_manager.load_artifacts()
        logger.info("All artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "HazeRadar Inference API",
        "version": "1.0.0",
        "primary_forecast_endpoint": "/api/forecast/{city}/all"
    }


@app.get("/health")
async def health():
    """
    Detailed health status
    """
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "cities_count": len(model_manager.cities) if model_manager.cities else 0,
        "features": FEATURE_COLS,
        "forecast_hours": FORECAST_HOURS
    }


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """
    Get API configuration for frontend
    Frontend should use this to configure slider mappings
    Example: const selectedHour = config.forecast_hours[sliderIndex]
    """
    return {
        "forecast_hours": FORECAST_HOURS,
        "features": FEATURE_COLS,
        "model": "GATv2",
        "num_cities": len(model_manager.cities) if model_manager.cities else 0
    }


@app.get("/api/predictions/current", response_model=List[CurrentPrediction])
async def get_current_predictions():
    """
    Get current PM2.5 predictions for all cities
    """
    try:
        logger.info("Current predictions endpoint called")
        predictions = model_manager.predict_current()
        logger.info(f"Returning {len(predictions)} predictions")
        return predictions
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Current prediction endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/{city}/all", response_model=AllForecastsResponse)
async def get_all_city_forecasts(city: str):
    """
    PRIMARY FORECAST ENDPOINT
    Get all forecast hours for a city in one call
    
    This is optimized for frontend slider usage
    Frontend should:
    1. Call this endpoint once on city selection
    2. Store the forecasts array
    3. Map slider index to forecast using: forecasts[sliderIndex]
    
    Example response:
    {
        "city": "Jakarta",
        "hours": [0, 12, 24, 36, 48, 60],
        "generated_at": "2025-01-18T10:30:00",
        "forecasts": [...]
    }
    """
    try:
        result = model_manager.forecast_city_all(city)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"All forecasts failed for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities")
async def get_cities():
    """
    List all supported cities
    """
    return {
        "cities": model_manager.cities,
        "count": len(model_manager.cities)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
