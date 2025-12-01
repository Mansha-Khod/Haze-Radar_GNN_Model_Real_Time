"""
HazeRadar Real-Time API Backend (Railway Deployment)
====================================================
Flagship version with full features:
- Real-time GNN predictions
- 24-hour forecast
- Uncertainty estimates
- AQI categorization
- NASA FIRMS fire data integration
- Supabase database support
- Auto and manual update
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

# -------------------- Configuration --------------------
class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
    MODEL_PATH = os.getenv("MODEL_PATH", "real_time_haze_infer.pt")
    GRAPH_CACHE = "city_graph_cache.json"
    NORMALIZATION_STATS = "normalization_stats.json"
    PORT = int(os.getenv("PORT", 8000))
    UPDATE_INTERVAL = 1800  # seconds (30 mins)
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
        logger.info("📍 Initializing data pipeline...")
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
                    logger.info("✅ Loaded graph cache")
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
        logger.info("💾 Saved graph cache")

    def _build_from_database(self):
        logger.info("📦 Building graph from database...")
        # Load cities
        response = self.supabase.table("gnn_training_data").select("city, latitude, longitude").execute()
        df = pd.DataFrame(response.data).drop_duplicates(subset=['city']).reset_index(drop=True)
        self.cities_df = df
        self.city_to_idx = {city: idx for idx, city in enumerate(df['city'])}

        # Load edges
        edges = []
        try:
            response = self.supabase.table("city_graph_structure").select("*").execute()
            graph_df = pd.DataFrame(response.data)
            for _, row in graph_df.iterrows():
                src = row['city']
                if src not in self.city_to_idx:
                    continue
                src_idx = self.city_to_idx[src]
                connected = eval(row['connected_cities']) if isinstance(row['connected_cities'], str) else row['connected_cities']
                for dst in connected:
                    dst = dst.strip() if isinstance(dst, str) else dst
                    if dst in self.city_to_idx:
                        edges.append([src_idx, self.city_to_idx[dst]])
        except Exception as e:
            logger.warning(f"No database edges: {e}")
        if len(edges) == 0:
            edges = self._build_proximity_graph()
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(f"Graph has {self.edge_index.shape[1]} edges")

        # Compute normalization stats
        train_resp = self.supabase.table("gnn_training_data").select("*").limit(1000).execute()
        train_df = pd.DataFrame(train_resp.data)
        for col in self.config.FEATURE_COLS:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        train_df = train_df.dropna()
        features = torch.tensor(train_df[self.config.FEATURE_COLS].values, dtype=torch.float32)
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

    def fetch_nasa_fires(self, region="south-east-asia") -> pd.DataFrame:
        if not self.config.FIRMS_API_KEY:
            logger.warning("⚠ No FIRMS API key, using empty fire data")
            return pd.DataFrame()
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.config.FIRMS_API_KEY}/VIIRS_SNPP_NRT/{region}/1"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                logger.info(f"🔥 Fetched {len(df)} fires from NASA FIRMS")
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"NASA FIRMS fetch error: {e}")
            return pd.DataFrame()

    def calculate_fire_features(self, fires_df: pd.DataFrame) -> Dict:
        fire_features = {}
        for city, idx in self.city_to_idx.items():
            city_data = self.cities_df[self.cities_df['city'] == city].iloc[0]
            lat, lon = city_data['latitude'], city_data['longitude']
            if not fires_df.empty:
                fires_df['distance'] = np.sqrt((fires_df['latitude']-lat)**2 + (fires_df['longitude']-lon)**2)*111
                nearby = fires_df[fires_df['distance']<100]
                fire_features[city] = {
                    'fire_confidence': nearby['confidence'].mean() if len(nearby) else 0,
                    'upwind_count': len(nearby)
                }
            else:
                fire_features[city] = {'fire_confidence': 0, 'upwind_count': 0}
        return fire_features

    def prepare_realtime_features(self):
        fires_df = self.fetch_nasa_fires()
        fire_features = self.calculate_fire_features(fires_df)
        weather_resp = self.supabase.table("gnn_training_data").select(
            "city, temperature, humidity, wind_speed, wind_direction, current_aqi"
        ).order("timestamp", desc=True).limit(len(self.city_to_idx)).execute()
        weather_df = pd.DataFrame(weather_resp.data)
        features = []
        for city in self.cities_df['city']:
            w = weather_df[weather_df['city']==city]
            if len(w)==0:
                features.append([25,70,5,90,0,0,1000,50])
                continue
            w = w.iloc[0]
            fire_feat = fire_features.get(city, {'fire_confidence':0,'upwind_count':0})
            features.append([
                float(w.get('temperature',25)),
                float(w.get('humidity',70)),
                float(w.get('wind_speed',5)),
                float(w.get('wind_direction',90)),
                float(fire_feat['fire_confidence']),
                float(fire_feat['upwind_count']),
                1000.0,
                float(w.get('current_aqi',50))
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
            city_data = self.pipeline.cities_df[self.pipeline.cities_df['city']==city].iloc[0]
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
            X = X*(1+0.01*hour*torch.randn_like(X)*0.1)  # temporal drift
            X, edge_index = X.to(self.device), edge_index.to(self.device)
            with torch.no_grad():
                pred, uncertainty = self.model(X, edge_index)
            for idx, city in enumerate(self.pipeline.cities_df['city']):
                city_data = self.pipeline.cities_df[self.pipeline.cities_df['city']==city].iloc[0]
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
app = FastAPI(title="HazeRadar API", version="1.0.0",
              description="Flagship real-time haze prediction using GNNs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

pipeline = None
predictor = None
scheduler = BackgroundScheduler()
device = torch.device("cpu")

# -------------------- Background Update --------------------
def update_predictions():
    global predictor
    try:
        logger.info("🔄 Updating predictions...")
        preds = predictor.predict_current()
        try:
            pipeline.supabase.table("realtime_predictions").upsert(preds).execute()
            logger.info(f"✅ Updated {len(preds)} predictions")
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

# -------------------- API Endpoints --------------------
@app.on_event("startup")
async def startup_event():
    global pipeline, predictor
    try:
        pipeline = DataPipeline(config)
        pipeline.initialize()
        model = RealtimeHazeGNN(in_feats=len(config.FEATURE_COLS)).to(device)
        if os.path.exists(config.MODEL_PATH):
            model.load_state_dict(torch.load(config.MODEL_PATH,map_location=device))
            logger.info("✅ Model loaded successfully")
        predictor = PredictionEngine(model, pipeline, device)
        update_predictions()
        scheduler.add_job(update_predictions,'interval',seconds=config.UPDATE_INTERVAL)
        scheduler.start()
        logger.info("🚀 API ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "last_update": predictor.last_update.isoformat() if predictor and predictor.last_update else None,
        "cities_count": len(pipeline.city_to_idx) if pipeline else 0,
        "version": "1.0.0"
    }

@app.get("/api/predictions/current", response_model=List[PredictionResponse])
async def get_current_predictions():
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    if predictor.last_predictions and predictor.last_update and (datetime.now()-predictor.last_update).seconds<300:
        return predictor.last_predictions
    return predictor.predict_current()

@app.get("/api/predictions/city/{city_name}", response_model=PredictionResponse)
async def get_city_prediction(city_name: str):
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    preds = predictor.last_predictions or predictor.predict_current()
    for p in preds:
        if p['city'].lower()==city_name.lower(): return p
    raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")

@app.get("/api/forecast/24h", response_model=List[ForecastResponse])
async def get_24h_forecast():
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    return predictor.forecast_24h(predictor.last_predictions or predictor.predict_current())

@app.get("/api/forecast/city/{city_name}", response_model=List[ForecastResponse])
async def get_city_forecast(city_name: str):
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    fc = predictor.forecast_24h(predictor.last_predictions or predictor.predict_current())
    city_fc = [f for f in fc if f['city'].lower()==city_name.lower()]
    if not city_fc:
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
    return city_fc

@app.post("/api/update")
async def manual_update(background_tasks: BackgroundTasks):
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
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

# -------------------- Main Entry --------------------
if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
