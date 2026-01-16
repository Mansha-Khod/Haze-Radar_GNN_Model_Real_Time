"""
HazeRadar Real-Time API Backend (Railway Deployment)
FIXED VERSION - Corrected AQI calculation and category mapping
"""
from urllib.parse import unquote
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HazeRadarAPI")

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
    MODEL_PATH = os.getenv("MODEL_PATH", "realtime_haze_gnn_infer.pt")
    GRAPH_CACHE = "city_graph_cache.json"
    PORT = int(os.getenv("PORT", 8000))
    UPDATE_INTERVAL = 21600
    FEATURE_COLS = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'avg_fire_confidence', 'upwind_fire_count', 'population_density',
        'current_aqi'
    ]

config = Config()

def pm25_to_aqi(pm25: float) -> float:
    """
    Convert PM2.5 to AQI using EPA breakpoints
    FIXED: Ensures correct AQI calculation
    """
    pm25 = max(0, pm25)
    
    # EPA PM2.5 AQI breakpoints (Concentration Low, Concentration High, AQI Low, AQI High)
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
            # Linear interpolation formula
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)
    
    # If PM2.5 > 500.4, cap at 500
    if pm25 > 500.4:
        return 500.0
    
    return round(aqi, 1)

def pm25_to_category(pm25: float) -> str:
    """
    FIXED: Correct category boundaries based on PM2.5 values
    """
    pm25 = max(0, pm25)
    
    if pm25 <= 12.0:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def aqi_to_category(aqi: float) -> str:
    """
    FIXED: Helper function to get category from AQI value
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

def get_future_weather(lat: float, lon: float) -> Dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        f"&forecast_days=3"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "temperature_2m": data["hourly"]["temperature_2m"][:72],
            "relative_humidity_2m": data["hourly"]["relative_humidity_2m"][:72],
            "wind_speed_10m": data["hourly"]["wind_speed_10m"][:72],
            "wind_direction_10m": data["hourly"]["wind_direction_10m"][:72]
        }
    except Exception as e:
        logger.warning(f"Weather fetch failed for lat={lat}, lon={lon}: {e}")
        return {
            "temperature_2m": [25.0] * 72,
            "relative_humidity_2m": [70.0] * 72,
            "wind_speed_10m": [5.0] * 72,
            "wind_direction_10m": [90.0] * 72
        }

class RealtimeHazeGNN(torch.nn.Module):
    def __init__(self, in_feats, hidden=128, out_feats=1, num_heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_feats, hidden // num_heads, heads=num_heads, dropout=dropout)
        self.ln1 = LayerNorm(hidden)
        self.gat2 = GATv2Conv(hidden, hidden // num_heads, heads=num_heads, dropout=dropout)
        self.ln2 = LayerNorm(hidden)
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, out_feats)
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 4, out_feats),
            torch.nn.Softplus()
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(self.ln1(h))
        h = self.dropout(h)
        h2 = self.gat2(h, edge_index)
        h2 = self.ln2(h2)
        h = F.elu(h + h2)
        pred = self.pred_head(h)
        uncertainty = self.uncertainty_head(h)
        return pred, uncertainty

class DataPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.city_to_idx = {}
        self.edge_index = None
        self.cities_df = None
        self.feature_mean = None
        self.feature_std = None

    def initialize(self):
        logger.info("Initializing data pipeline...")
        self._load_cache()
        if self.edge_index is None or len(self.city_to_idx) == 0:
            self._build_from_database()
            self._save_cache()

    def _load_cache(self):
        if os.path.exists(self.config.GRAPH_CACHE):
            try:
                with open(self.config.GRAPH_CACHE, 'r') as f:
                    cache = json.load(f)
                    self.cities_df = pd.DataFrame(cache['cities'])
                    self.city_to_idx = cache['city_to_idx']
                    self.edge_index = torch.tensor(cache['edges'], dtype=torch.long).t().contiguous()
                    self.feature_mean = torch.tensor(cache['feature_mean'], dtype=torch.float32)
                    self.feature_std = torch.tensor(cache['feature_std'], dtype=torch.float32)
                    logger.info("Loaded graph cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        cache = {
            'cities': self.cities_df.to_dict('records'),
            'city_to_idx': self.city_to_idx,
            'edges': self.edge_index.t().tolist(),
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist()
        }
        with open(self.config.GRAPH_CACHE, 'w') as f:
            json.dump(cache, f)
        logger.info("Saved graph cache")

    def _build_from_database(self):
        logger.info("Building graph from Supabase...")
        response = self.supabase.table("gnn_training_data").select("*").execute()
        if not hasattr(response, "data") or response.data is None:
            raise ValueError("Supabase fetch failed")
        raw = pd.DataFrame(response.data)
        if len(raw) > 0 and isinstance(raw.iloc[0]["city"], dict):
            raw["city"] = raw["city"].apply(lambda x: x.get("city") if isinstance(x, dict) else x)

        df = raw.drop_duplicates(subset=["city"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No data found in gnn_training_data")
        self.cities_df = df
        self.city_to_idx = {city: idx for idx, city in enumerate(df['city'])}

        edges = self._build_proximity_graph()
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(f"Graph has {self.edge_index.shape[1]} edges")

        features = torch.tensor(df[self.config.FEATURE_COLS].fillna(0).values, dtype=torch.float32)
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0) + 1e-6

    def _build_proximity_graph(self):
        from math import radians, cos, sin, asin, sqrt
        edges = []
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return 6371 * c
        MAX_DIST = 300
        for i, row1 in self.cities_df.iterrows():
            for j, row2 in self.cities_df.iterrows():
                if i == j:
                    continue
                if haversine(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude']) <= MAX_DIST:
                    edges.append([i, j])
        if not edges:
            edges = [[i, i] for i in range(len(self.cities_df))]
        return edges

    def prepare_realtime_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weather_resp = self.supabase.table("gnn_training_data").select(
            "city, temperature, humidity, wind_speed, wind_direction, avg_fire_confidence, upwind_fire_count, population_density, current_aqi"
        ).order("timestamp", desc=True).limit(len(self.city_to_idx)).execute()
    
        if not hasattr(weather_resp, "data") or weather_resp.data is None:
            raise ValueError("Failed to fetch realtime features from Supabase")
    
        df = pd.DataFrame(weather_resp.data)
    
        for col in ["temperature", "humidity", "wind_speed", "wind_direction", 
                    "avg_fire_confidence", "upwind_fire_count", 
                    "population_density", "current_aqi"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
        features = []
        raw_features = []
        for city in self.cities_df['city']:
            row = df[df["city"] == city]
            if len(row) == 0:
                features.append([25, 70, 5, 90, 0, 0, 1000, 50])
                raw_features.append([25, 70, 5, 90, 0, 0, 1000, 50])
                continue
    
            row = row.iloc[0]
            feature_vector = [
                row.temperature, row.humidity, row.wind_speed, row.wind_direction,
                row.avg_fire_confidence, row.upwind_fire_count,
                row.population_density, row.current_aqi
            ]
            features.append(feature_vector)
            raw_features.append(feature_vector)
    
        X_raw = torch.tensor(raw_features, dtype=torch.float32)
        X = torch.tensor(features, dtype=torch.float32)
        X = (X - self.feature_mean) / self.feature_std
        return X, self.edge_index, X_raw

def build_72h_forecast(
    model,
    pipeline: DataPipeline,
    city: str,
    weather: Dict,
    base_pm25: float,
    device: torch.device,
    predictor
) -> List[Dict]:
    """
    FIXED: Build 72-hour forecast with proper PM2.5 bounds and AQI calculation
    """
    forecast = []

    for key in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]:
        if len(weather[key]) < 72:
            weather[key] += [weather[key][-1]] * (72 - len(weather[key]))

    if predictor.last_predictions is None:
        raise RuntimeError("Predictions not initialized")

    current_preds = predictor.last_predictions
    current_map = {p["city"]: p for p in current_preds}
    city_data = pipeline.cities_df[pipeline.cities_df["city"] == city].iloc[0]

    # FIXED: Ensure base_pm25 is within realistic bounds
    prev_pm25 = max(5.0, min(150.0, base_pm25))
    
    neighbor_pm25_map = {}
    for p in current_preds:
        neighbor_pm25_map[p["city"]] = p["predicted_pm25"]

    DEFAULT_TEMP = 25.0
    DEFAULT_HUM = 70.0
    DEFAULT_WS = 5.0
    DEFAULT_WD = 90.0
    DEFAULT_POP = 1000

    for t in range(72):
        temp = weather["temperature_2m"][t]
        humidity = weather["relative_humidity_2m"][t]
        wind_speed = weather["wind_speed_10m"][t]
        wind_dir = weather["wind_direction_10m"][t]

        all_features = []

        for _, row in pipeline.cities_df.iterrows():
            live = current_map.get(row["city"])

            if row["city"] == city:
                # FIXED: Calculate AQI from PM2.5, not the other way around
                current_aqi = pm25_to_aqi(prev_pm25)
                
                if live:
                    fire = live.get("avg_fire_confidence", 0)
                    upwind = live.get("upwind_fire_count", 0)
                    pop = live.get("population_density", DEFAULT_POP)
                else:
                    fire, upwind, pop = 0, 0, DEFAULT_POP

                all_features.append([temp, humidity, wind_speed, wind_dir, fire, upwind, pop, current_aqi])

            else:
                if live:
                    fire = live.get("avg_fire_confidence", 0)
                    upwind = live.get("upwind_fire_count", 0)
                    pop = live.get("population_density", DEFAULT_POP)
                    
                    neighbor_pm25 = neighbor_pm25_map.get(row["city"], live["predicted_pm25"])
                    neighbor_pm25 = 0.98 * neighbor_pm25 + 0.02 * prev_pm25
                    neighbor_pm25_map[row["city"]] = neighbor_pm25
                    
                    neighbor_aqi = pm25_to_aqi(neighbor_pm25)
                    temp2 = live.get("temperature", DEFAULT_TEMP)
                    hum2 = live.get("humidity", DEFAULT_HUM)
                    ws2 = live.get("wind_speed", DEFAULT_WS)
                    wd2 = live.get("wind_direction", DEFAULT_WD)
                else:
                    fire, upwind, pop = 0, 0, DEFAULT_POP
                    neighbor_aqi = 50
                    temp2 = DEFAULT_TEMP
                    hum2 = DEFAULT_HUM
                    ws2 = DEFAULT_WS
                    wd2 = DEFAULT_WD

                all_features.append([temp2, hum2, ws2, wd2, fire, upwind, pop, neighbor_aqi])

        X = torch.tensor(all_features, dtype=torch.float32)
        X = (X - pipeline.feature_mean) / pipeline.feature_std
        X = X.to(device)

        with torch.no_grad():
            pred, uncertainty = model(X, pipeline.edge_index)

        city_idx = pipeline.city_to_idx[city]
        
        raw_pm25 = float(pred[city_idx].cpu().numpy())
        
        # FIXED: Apply realistic PM2.5 bounds (5-150 for typical conditions)
        raw_pm25 = max(5.0, min(150.0, raw_pm25))
        
        # FIXED: More conservative temporal smoothing
        pm25 = 0.7 * prev_pm25 + 0.3 * raw_pm25
        pm25 = max(5.0, min(150.0, pm25))
        
        unc = float(uncertainty[city_idx].cpu().numpy())
        
        # FIXED: Calculate AQI from PM2.5
        aqi = pm25_to_aqi(pm25)
        category = pm25_to_category(pm25)
        
        if t % 12 == 0:
            logger.info(f"{city} t={t}h | PM2.5={pm25:.1f} | AQI={aqi:.1f} | {category}")

        forecast.append({
            "hour": t,
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "pm25": round(pm25, 2),
            "aqi": round(aqi, 1),
            "uncertainty": round(unc, 2),
            "category": category,
            "timestamp": (datetime.now() + timedelta(hours=t)).isoformat()
        })

        prev_pm25 = pm25

    return forecast

class PredictionEngine:
    def __init__(self, model, pipeline: DataPipeline, device: torch.device):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.model.eval()
        self.last_predictions = None
        self.last_update = None

    def predict_current(self) -> List[Dict]:
        """
        FIXED: Predict current conditions with proper PM2.5 and AQI handling
        """
        X, edge_index, X_raw = self.pipeline.prepare_realtime_features()
        X = X.to(self.device)

        with torch.no_grad():
            pred, uncertainty = self.model(X, edge_index)

        results = []
        for idx, city in enumerate(self.pipeline.cities_df['city']):
            city_data = self.pipeline.cities_df.iloc[idx]
            
            # Model outputs PM2.5
            pm25 = float(pred[idx].cpu().numpy())
            
            # FIXED: Apply realistic bounds
            pm25 = max(5.0, min(150.0, pm25))
            
            unc = float(uncertainty[idx].cpu().numpy())
            
            # FIXED: Calculate AQI from PM2.5, get category from PM2.5
            aqi = pm25_to_aqi(pm25)
            category = pm25_to_category(pm25)

            results.append({
                'city': city,
                'latitude': float(city_data['latitude']),
                'longitude': float(city_data['longitude']),
                'temperature': float(X_raw[idx][0]),
                'humidity': float(X_raw[idx][1]),
                'wind_speed': float(X_raw[idx][2]),
                'wind_direction': float(X_raw[idx][3]),
                'avg_fire_confidence': float(X_raw[idx][4]),
                'upwind_fire_count': float(X_raw[idx][5]),
                'population_density': float(X_raw[idx][6]),
                'predicted_pm25': round(pm25, 2),
                'uncertainty': round(unc, 2),
                'aqi': round(aqi, 1),
                'aqi_category': category,
                'timestamp': datetime.now().isoformat()
            })

        self.last_predictions = results
        self.last_update = datetime.now()
        
        for r in results:
            logger.info(
                "✅ %s | PM2.5=%.1f | AQI=%.0f | %s",
                r["city"],
                r["predicted_pm25"],
                r["aqi"],
                r["aqi_category"],
            )
        return results

app = FastAPI(title="HazeRadar API", version="2.0.1-fixed")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

pipeline = None
predictor = None
forecast_cache = {}
scheduler = BackgroundScheduler()
device = torch.device("cpu")

def update_predictions():
    global predictor, forecast_cache
    try:
        logger.info("🔄 Updating predictions...")
        preds = predictor.predict_current()

        new_cache = {}
        for _, row in pipeline.cities_df.iterrows():
            city = row["city"]
            lat, lon = row["latitude"], row["longitude"]

            weather = get_future_weather(lat, lon)
            base_pm25 = next((p["predicted_pm25"] for p in preds if p["city"] == city), 40.0)

            full = build_72h_forecast(
                predictor.model,
                pipeline,
                city,
                weather,
                base_pm25,
                device,
                predictor
            )
            new_cache[city.lower()] = full

        forecast_cache.clear()
        forecast_cache.update(new_cache)
        logger.info(f"✅ Updated forecasts for {len(new_cache)} cities")

    except Exception as e:
        logger.error(f"❌ Prediction update failed: {e}", exc_info=True)

class PredictionResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_pm25: float
    uncertainty: float
    aqi: float
    aqi_category: str
    timestamp: str

class ForecastHourResponse(BaseModel):
    hour: int
    temperature: float
    humidity: float
    wind_speed: float
    pm25: float
    aqi: float
    uncertainty: float
    category: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_update: Optional[str]
    cities_count: int
    version: str

@app.on_event("startup")
async def startup_event():
    global pipeline, predictor
    pipeline = DataPipeline(config)
    pipeline.initialize()
    pipeline.edge_index = pipeline.edge_index.to(device)
    
    model = RealtimeHazeGNN(in_feats=len(config.FEATURE_COLS)).to(device)
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning(f"⚠️ Model file not found at {config.MODEL_PATH}")
    
    predictor = PredictionEngine(model, pipeline, device)
    update_predictions()
    scheduler.add_job(update_predictions, 'interval', seconds=config.UPDATE_INTERVAL)
    scheduler.start()
    logger.info("🚀 API ready!")

@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "last_update": predictor.last_update.isoformat() if predictor.last_update else None,
        "cities_count": len(pipeline.city_to_idx),
        "version": "2.0.1-fixed"
    }

@app.get("/api/predictions/current", response_model=List[PredictionResponse])
async def get_current_predictions():
    if predictor.last_predictions is None:
        raise HTTPException(503, "Predictions not ready yet")
    return predictor.last_predictions

@app.get("/api/predictions/city/{city_name}", response_model=PredictionResponse)
async def get_city_prediction(city_name: str):
    if predictor.last_predictions is None:
        raise HTTPException(503, "Predictions not ready yet")
    preds = predictor.last_predictions

    for p in preds:
        if p['city'].lower() == city_name.lower():
            return p
    raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

@app.get("/api/forecast/{city}", response_model=List[ForecastHourResponse])
async def forecast_city(city: str):
    city = unquote(city)
    city_match = pipeline.cities_df[pipeline.cities_df["city"].str.lower() == city.lower()]
    if city_match.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")

    if predictor.last_predictions is None:
        raise HTTPException(503, "Predictions not ready yet")

    city_key = city.lower()
    if city_key not in forecast_cache:
        raise HTTPException(503, "Forecast not ready yet")
    
    forecast_data = forecast_cache[city_key]
    slider_indices = [0, 12, 24, 36, 48, 60]
    slider_data = []
    
    for idx in slider_indices:
        if idx < len(forecast_data):
            point = dict(forecast_data[idx])
            point['hour'] = idx
            slider_data.append(point)
    
    return slider_data

@app.post("/api/update")
async def manual_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_predictions)
    return {"status": "Update triggered"}

@app.get("/api/stats")
async def stats():
    return {
        "cities_count": len(pipeline.city_to_idx),
        "last_update": predictor.last_update.isoformat() if predictor.last_update else None,
        "graph_edges": pipeline.edge_index.shape[1],
        "model_params": sum(p.numel() for p in predictor.model.parameters())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT)
