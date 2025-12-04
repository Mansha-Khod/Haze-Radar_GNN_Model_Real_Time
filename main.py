"""
HazeRadar Real-Time API Backend (Railway Deployment)
---------------------------------------------------
Corrected to use `gnn_training_data` table and 3-hour interval batching
"""

import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from supabase import create_client

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HazeRadarAPI")

# -------------------- Config --------------------
class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
    MODEL_PATH = os.getenv("MODEL_PATH", "real_time_haze_infer.pt")
    GRAPH_CACHE = "city_graph_cache.json"
    NORMALIZATION_STATS = "normalization_stats.json"
    PORT = int(os.getenv("PORT", 8000))
    UPDATE_INTERVAL = 10800  # 3 hours = 10800 seconds
    FEATURE_COLS = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'avg_fire_confidence', 'upwind_fire_count', 'population_density',
        'current_aqi'
    ]

config = Config()

# -------------------- GNN Model --------------------
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

# -------------------- Data Pipeline --------------------
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
        logger.info(" Initializing data pipeline...")
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
        logger.info(" Building graph from Supabase...")
        response = self.supabase.table("gnn_training_data").select("*").execute()
        df = pd.DataFrame(response.data).drop_duplicates(subset=['city']).reset_index(drop=True)
        if df.empty:
            raise ValueError("No data found in gnn_training_data")
        self.cities_df = df
        self.city_to_idx = {city: idx for idx, city in enumerate(df['city'])}

        # Proximity graph for edges
        edges = self._build_proximity_graph()
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(f"Graph has {self.edge_index.shape[1]} edges")

        # Feature normalization
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
                if i == j: continue
                if haversine(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude']) <= MAX_DIST:
                    edges.append([i, j])
        if not edges:
            edges = [[i,i] for i in range(len(self.cities_df))]
        return edges

    def prepare_realtime_features(self):
        weather_resp = self.supabase.table("gnn_training_data").select(
            "city, temperature, humidity, wind_speed, wind_direction, avg_fire_confidence, upwind_fire_count, population_density, current_aqi"
        ).order("timestamp", desc=True).limit(len(self.city_to_idx)).execute()
    
        df = pd.DataFrame(weather_resp.data)
    
        for col in ["temperature", "humidity", "wind_speed", "wind_direction", 
                    "avg_fire_confidence", "upwind_fire_count", 
                    "population_density", "current_aqi"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
        features = []
        for city in self.cities_df['city']:
            row = df[df["city"] == city]
            if len(row) == 0:
                features.append([25,70,5,90,0,0,1000,50])  
                continue
    
            row = row.iloc[0]
            features.append([
                row.temperature, row.humidity, row.wind_speed, row.wind_direction,
                row.avg_fire_confidence, row.upwind_fire_count,
                row.population_density, row.current_aqi
            ])
    
        X = torch.tensor(features, dtype=torch.float32)
        X = (X - self.feature_mean) / self.feature_std
        return X, self.edge_index


# -------------------- Prediction Engine --------------------
class PredictionEngine:
    def __init__(self, model, pipeline: DataPipeline, device: torch.device):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.model.eval()
        self.last_predictions = None
        self.last_update = None

    def predict_current(self) -> List[Dict]:
        X, edge_index = self.pipeline.prepare_realtime_features()
        X, edge_index = X.to(self.device), edge_index.to(self.device)
        with torch.no_grad():
            pred, uncertainty = self.model(X, edge_index)
        results = []
        for idx, city in enumerate(self.pipeline.cities_df['city']):
            city_data = self.pipeline.cities_df.iloc[idx]
            pm25 = float(pred[idx].cpu().numpy())
            unc = float(uncertainty[idx].cpu().numpy()*100)
            results.append({
                'city': city,
                'latitude': float(city_data['latitude']),
                'longitude': float(city_data['longitude']),
                'predicted_pm25': max(0, pm25),
                'uncertainty': unc,
                'aqi_category': self._pm25_to_category(pm25),
                'timestamp': datetime.now().isoformat()
            })
        self.last_predictions = results
        self.last_update = datetime.now()
        return results

    def forecast_24h(self, current_pred: List[Dict]) -> List[Dict]:
        forecasts = []
        for hour in range(1,25):
            X, edge_index = self.pipeline.prepare_realtime_features()
            X = X*(1+0.01*hour*torch.randn_like(X)*0.1)
            X, edge_index = X.to(self.device), edge_index.to(self.device)
            with torch.no_grad():
                pred, uncertainty = self.model(X, edge_index)
            for idx, city in enumerate(self.pipeline.cities_df['city']):
                city_data = self.pipeline.cities_df.iloc[idx]
                pm25 = float(pred[idx].cpu().numpy())
                unc = float(uncertainty[idx].cpu().numpy()*100)
                forecasts.append({
                    'city': city,
                    'latitude': float(city_data['latitude']),
                    'longitude': float(city_data['longitude']),
                    'predicted_pm25': max(0, pm25),
                    'uncertainty': unc,
                    'forecast_hour': hour,
                    'aqi_category': self._pm25_to_category(pm25),
                    'timestamp': (datetime.now()+timedelta(hours=hour)).isoformat()
                })
        return forecasts

    @staticmethod
    def _pm25_to_category(pm25: float) -> str:
        if pm25<=12: return "Good"
        elif pm25<=35.4: return "Moderate"
        elif pm25<=55.4: return "Unhealthy for Sensitive"
        elif pm25<=150.4: return "Unhealthy"
        elif pm25<=250.4: return "Very Unhealthy"
        else: return "Hazardous"

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="HazeRadar API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

pipeline = None
predictor = None
scheduler = BackgroundScheduler()
device = torch.device("cpu")

def update_predictions():
    global predictor
    try:
        logger.info("🔄 Updating predictions...")
        preds = predictor.predict_current()
        try:
            pipeline.supabase.table("realtime_predictions").upsert(preds).execute()
            logger.info(f"Updated {len(preds)} predictions")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    except Exception as e:
        logger.error(f"Prediction update failed: {e}")

# -------------------- API Models --------------------
class PredictionResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_pm25: float
    uncertainty: float
    aqi_category: str
    timestamp: str

class ForecastResponse(PredictionResponse):
    forecast_hour: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_update: Optional[str]
    cities_count: int
    version: str

# -------------------- Endpoints --------------------
@app.on_event("startup")
async def startup_event():
    global pipeline, predictor
    pipeline = DataPipeline(config)
    pipeline.initialize()
    model = RealtimeHazeGNN(in_feats=len(config.FEATURE_COLS)).to(device)
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH,map_location=device))
        logger.info(" Model loaded successfully")
    predictor = PredictionEngine(model, pipeline, device)
    update_predictions()
    scheduler.add_job(update_predictions,'interval',seconds=config.UPDATE_INTERVAL)
    scheduler.start()
    logger.info(" API ready!")

@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "last_update": predictor.last_update.isoformat() if predictor.last_update else None,
        "cities_count": len(pipeline.city_to_idx),
        "version": "1.0.0"
    }

@app.get("/api/predictions/current", response_model=List[PredictionResponse])
async def get_current_predictions():
    return predictor.last_predictions or predictor.predict_current()

@app.get("/api/predictions/city/{city_name}", response_model=PredictionResponse)
async def get_city_prediction(city_name: str):
    preds = predictor.last_predictions or predictor.predict_current()
    for p in preds:
        if p['city'].lower() == city_name.lower(): return p
    raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

@app.get("/api/forecast/city/{city_name}", response_model=List[ForecastResponse])
async def get_city_forecast(city_name: str):
    fc = predictor.forecast_24h(predictor.last_predictions or predictor.predict_current())
    city_fc = [f for f in fc if f['city'].lower()==city_name.lower()]
    if not city_fc:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
    return city_fc

@app.post("/api/update")
async def manual_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_predictions)
    return {"status":"Update triggered"}

@app.get("/api/stats")
async def stats():
    return {
        "cities_count": len(pipeline.city_to_idx),
        "last_update": predictor.last_update.isoformat() if predictor.last_update else None,
        "graph_edges": predictor.pipeline.edge_index.shape[1],
        "model_params": sum(p.numel() for p in predictor.model.parameters())
    }
from fastapi import Query
import requests

@app.get("/predict_future")
async def predict_future(city: str = Query(...), hours: int = Query(3)):
    """
    Predict PM2.5 for a future time using forecast weather input.
    Hours: 1-24 by default
    """

    try:
        
        row = self.cities_df[self.cities_df["city"] == city].iloc[0]
        lat, lon = row["latitude"], row["longitude"]

        
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,"
            f"windspeed_10m,winddirection_10m&forecast_days=2"
        )
        resp = requests.get(forecast_url).json()

        index = hours 

        
        features = [
            resp["hourly"]["temperature_2m"][index],
            resp["hourly"]["relativehumidity_2m"][index],
            resp["hourly"]["windspeed_10m"][index],
            resp["hourly"]["winddirection_10m"][index],
            0,   
            0,   
            row["population_density"],
            row["current_aqi"]
        ]

        
        X = torch.tensor([features], dtype=torch.float32)
        X = (X - self.feature_mean) / self.feature_std

        
        with torch.no_grad():
            pred = self.model(X, self.edge_index)[0].item()

        return {
            "city": city,
            "hours_ahead": hours,
            "predicted_pm25": round(pred, 3),
            "latitude": lat,
            "longitude": lon
        }

    except Exception as e:
        return {"error": str(e)}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
