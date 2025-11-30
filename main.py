"""
HazeRadar Real-Time API Backend (Railway Deployment)
====================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client, Client
import requests
import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (from environment variables)
# ============================================================================

class Config:
    # Supabase (set these in Railway environment variables)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://daxrnmvkpikjvvzgrhko.supabase.co/")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRheHJubXZrcGlranZ2emdyaGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2OTkyNjEsImV4cCI6MjA3NjI3NTI2MX0.XWJ_aWUh5Eci5tQSRAATqDXmQ5nh2eHQGzYu6qMcsvQ")
    
    # NASA FIRMS API
    FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
    
    # Model path (will be in repo)
    MODEL_PATH = os.getenv("MODEL_PATH", "realtime_haze_gnn.pt")
    GRAPH_CACHE = "city_graph_cache.json"
    
    # API settings
    PORT = int(os.getenv("PORT", 8000))
    UPDATE_INTERVAL = 1800  # 30 minutes in seconds
    
    # Feature columns
    FEATURE_COLS = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'avg_fire_confidence', 'upwind_fire_count', 
        'population_density', 'current_aqi'
    ]

# ============================================================================
# GNN MODEL (same architecture as training)
# ============================================================================

class RealtimeHazeGNN(torch.nn.Module):
    def __init__(self, in_feats: int, hidden: int, out_feats: int, 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.gat1 = GATv2Conv(in_feats, hidden // num_heads, heads=num_heads, 
                              dropout=dropout, concat=True)
        self.ln1 = LayerNorm(hidden)
        
        self.gat2 = GATv2Conv(hidden, hidden // num_heads, heads=num_heads, 
                              dropout=dropout, concat=True)
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
        h = self.ln1(h)
        h = F.elu(h)
        h = self.dropout(h)
        
        h2 = self.gat2(h, edge_index)
        h2 = self.ln2(h2)
        h = h + h2
        h = F.elu(h)
        
        pred = self.pred_head(h)
        uncertainty = self.uncertainty_head(h)
        
        return pred, uncertainty

# ============================================================================
# DATA PIPELINE
# ============================================================================

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
        """Load city graph structure from cache or database"""
        logger.info("📍 Initializing data pipeline...")
        
        # Try loading from cache first
        if os.path.exists(self.config.GRAPH_CACHE):
            logger.info("Loading from cache...")
            with open(self.config.GRAPH_CACHE, 'r') as f:
                cache = json.load(f)
                self.cities_df = pd.DataFrame(cache['cities'])
                self.city_to_idx = cache['city_to_idx']
                self.edge_index = torch.tensor(cache['edges'], dtype=torch.long)
                self.feature_mean = torch.tensor(cache['feature_mean'], dtype=torch.float32)
                self.feature_std = torch.tensor(cache['feature_std'], dtype=torch.float32)
        else:
            logger.info("Building graph from database...")
            self._build_from_database()
            self._save_cache()
        
        logger.info(f"✓ Loaded {len(self.city_to_idx)} cities, {self.edge_index.shape[1]} edges")
    
    def _build_from_database(self):
        """Build graph structure from Supabase"""
        # Load cities
        response = self.supabase.table("gnn_training_data").select("city, latitude, longitude").execute()
        df = pd.DataFrame(response.data)
        self.cities_df = df.drop_duplicates(subset=['city']).reset_index(drop=True)
        self.city_to_idx = {city: idx for idx, city in enumerate(self.cities_df['city'])}
        
        # Try to build edge index from database
        edges = []
        try:
            response = self.supabase.table("city_graph_structure").select("*").execute()
            graph_df = pd.DataFrame(response.data)
            
            for _, row in graph_df.iterrows():
                src_city = row['city']
                if src_city not in self.city_to_idx:
                    continue
                
                src_idx = self.city_to_idx[src_city]
                
                try:
                    if isinstance(row['connected_cities'], str):
                        connected = eval(row['connected_cities'])
                    else:
                        connected = row['connected_cities']
                except:
                    continue
                
                for dst_city in connected:
                    if isinstance(dst_city, str):
                        dst_city = dst_city.strip()
                    if dst_city in self.city_to_idx:
                        dst_idx = self.city_to_idx[dst_city]
                        edges.append([src_idx, dst_idx])
        except Exception as e:
            logger.warning(f"Could not load graph structure: {e}")
        
        # If no edges found, build automatically based on proximity
        if len(edges) == 0:
            logger.info("Building graph from spatial proximity...")
            edges = self._build_proximity_graph()
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(f"Built graph with {self.edge_index.shape[1]} edges")
    
    def _build_proximity_graph(self) -> list:
        """Build graph edges based on geographic proximity"""
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km"""
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return 6371 * c
        
        MAX_DISTANCE_KM = 300  # Connect cities within 300km
        edges = []
        
        for idx1, row1 in self.cities_df.iterrows():
            for idx2, row2 in self.cities_df.iterrows():
                if idx1 == idx2:
                    continue
                
                distance = haversine_distance(
                    row1['latitude'], row1['longitude'],
                    row2['latitude'], row2['longitude']
                )
                
                if distance <= MAX_DISTANCE_KM:
                    edges.append([idx1, idx2])
        
        # If still no edges, create self-loops
        if len(edges) == 0:
            logger.warning("Creating self-loops as fallback")
            edges = [[i, i] for i in range(len(self.cities_df))]
        
        return edges
        
        # Calculate normalization stats from training data
        train_response = self.supabase.table("gnn_training_data").select("*").limit(1000).execute()
        train_df = pd.DataFrame(train_response.data)
        for col in self.config.FEATURE_COLS:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        train_df = train_df.dropna()
        
        features = torch.tensor(train_df[self.config.FEATURE_COLS].values, dtype=torch.float32)
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0) + 1e-6
    
    def _save_cache(self):
        """Save graph to cache file"""
        cache = {
            'cities': self.cities_df.to_dict('records'),
            'city_to_idx': self.city_to_idx,
            'edges': self.edge_index.t().tolist(),
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist()
        }
        with open(self.config.GRAPH_CACHE, 'w') as f:
            json.dump(cache, f)
        logger.info("✓ Saved graph cache")
    
    def fetch_nasa_fires(self, region: str = "south-east-asia") -> pd.DataFrame:
        """Fetch latest fire hotspots from NASA FIRMS"""
        if not self.config.FIRMS_API_KEY:
            logger.warning("⚠ No FIRMS API key, using cached fire data")
            return pd.DataFrame()
        
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.config.FIRMS_API_KEY}/VIIRS_SNPP_NRT/{region}/1"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"🔥 Fetched {len(df)} active fires from NASA FIRMS")
                return df
            else:
                logger.warning(f"⚠ NASA FIRMS API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch NASA data: {e}")
            return pd.DataFrame()
    
    def calculate_fire_features(self, fires_df: pd.DataFrame) -> Dict:
        """Calculate fire-related features for each city"""
        if fires_df.empty:
            return {city: {'fire_confidence': 0, 'upwind_count': 0} 
                    for city in self.city_to_idx.keys()}
        
        fire_features = {}
        
        for city, idx in self.city_to_idx.items():
            city_data = self.cities_df[self.cities_df['city'] == city].iloc[0]
            lat, lon = city_data['latitude'], city_data['longitude']
            
            # Calculate distances
            fires_df['distance'] = np.sqrt(
                (fires_df['latitude'] - lat)**2 + 
                (fires_df['longitude'] - lon)**2
            ) * 111
            
            nearby = fires_df[fires_df['distance'] < 100]
            
            fire_features[city] = {
                'fire_confidence': nearby['confidence'].mean() if len(nearby) > 0 else 0,
                'upwind_count': len(nearby)
            }
        
        return fire_features
    
    def prepare_realtime_features(self) -> tuple:
        """Prepare current features for prediction"""
        # Fetch data
        fires_df = self.fetch_nasa_fires()
        fire_features = self.calculate_fire_features(fires_df)
        
        # Get latest weather from database (fallback)
        weather_response = self.supabase.table("gnn_training_data")\
            .select("city, temperature, humidity, wind_speed, wind_direction, current_aqi")\
            .order("timestamp", desc=True)\
            .limit(len(self.city_to_idx)).execute()
        weather_df = pd.DataFrame(weather_response.data)
        
        # Build feature matrix
        features = []
        for city in self.cities_df['city']:
            weather = weather_df[weather_df['city'] == city]
            if len(weather) == 0:
                features.append([25, 70, 5, 90, 0, 0, 1000, 50])
                continue
            
            weather = weather.iloc[0]
            fire_feat = fire_features.get(city, {'fire_confidence': 0, 'upwind_count': 0})
            
            features.append([
                float(weather.get('temperature', 25)),
                float(weather.get('humidity', 70)),
                float(weather.get('wind_speed', 5)),
                float(weather.get('wind_direction', 90)),
                float(fire_feat['fire_confidence']),
                float(fire_feat['upwind_count']),
                1000.0,
                float(weather.get('current_aqi', 50))
            ])
        
        X = torch.tensor(features, dtype=torch.float32)
        
        # Normalize
        X = (X - self.feature_mean) / self.feature_std
        
        return X, self.edge_index

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    def __init__(self, model, pipeline: DataPipeline, device: torch.device):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.model.eval()
        self.last_predictions = None
        self.last_update = None
    
    def predict_current(self) -> List[Dict]:
        """Generate current predictions"""
        X, edge_index = self.pipeline.prepare_realtime_features()
        X = X.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            pred, uncertainty = self.model(X, edge_index)
        
        # Build results
        results = []
        for idx, city in enumerate(self.pipeline.cities_df['city']):
            city_data = self.pipeline.cities_df[self.pipeline.cities_df['city'] == city].iloc[0]
            
            pm25 = float(pred[idx].cpu().numpy())
            unc = float(uncertainty[idx].cpu().numpy() * 100)
            
            results.append({
                'city': city,
                'latitude': float(city_data['latitude']),
                'longitude': float(city_data['longitude']),
                'predicted_pm25': max(0, pm25),  # Ensure non-negative
                'uncertainty': unc,
                'aqi_category': self._pm25_to_category(pm25),
                'timestamp': datetime.now().isoformat()
            })
        
        self.last_predictions = results
        self.last_update = datetime.now()
        
        return results
    
    def forecast_24h(self, current_pred: List[Dict]) -> List[Dict]:
        """Generate 24-hour forecast"""
        forecasts = []
        
        for hour in range(1, 25):
            X, edge_index = self.pipeline.prepare_realtime_features()
            
            # Add temporal drift (simplified)
            X = X * (1 + 0.01 * hour * torch.randn_like(X) * 0.1)
            
            X = X.to(self.device)
            edge_index = edge_index.to(self.device)
            
            with torch.no_grad():
                pred, uncertainty = self.model(X, edge_index)
            
            for idx, city in enumerate(self.pipeline.cities_df['city']):
                city_data = self.pipeline.cities_df[self.pipeline.cities_df['city'] == city].iloc[0]
                
                pm25 = float(pred[idx].cpu().numpy())
                unc = float(uncertainty[idx].cpu().numpy() * 100)
                
                forecasts.append({
                    'city': city,
                    'latitude': float(city_data['latitude']),
                    'longitude': float(city_data['longitude']),
                    'predicted_pm25': max(0, pm25),
                    'uncertainty': unc,
                    'forecast_hour': hour,
                    'aqi_category': self._pm25_to_category(pm25),
                    'timestamp': (datetime.now() + timedelta(hours=hour)).isoformat()
                })
        
        return forecasts
    
    @staticmethod
    def _pm25_to_category(pm25: float) -> str:
        if pm25 <= 12: return "Good"
        elif pm25 <= 35.4: return "Moderate"
        elif pm25 <= 55.4: return "Unhealthy for Sensitive"
        elif pm25 <= 150.4: return "Unhealthy"
        elif pm25 <= 250.4: return "Very Unhealthy"
        else: return "Hazardous"

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="HazeRadar API",
    description="Real-time haze prediction using Graph Neural Networks",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = Config()
device = torch.device('cpu')  # Railway uses CPU
pipeline = None
predictor = None
scheduler = BackgroundScheduler()

# ============================================================================
# API MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_pm25: float
    uncertainty: float
    aqi_category: str
    timestamp: str

class ForecastResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_pm25: float
    uncertainty: float
    forecast_hour: int
    aqi_category: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_update: Optional[str]
    cities_count: int
    version: str

# ============================================================================
# BACKGROUND TASK
# ============================================================================

def update_predictions():
    """Background task to update predictions every 30 minutes"""
    global predictor
    
    try:
        logger.info("🔄 Updating predictions...")
        predictions = predictor.predict_current()
        
        # Save to database
        try:
            pipeline.supabase.table("realtime_predictions").upsert(predictions).execute()
            logger.info(f"✅ Updated {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            
    except Exception as e:
        logger.error(f"Prediction update failed: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and pipeline on startup"""
    global pipeline, predictor
    
    logger.info("🚀 Starting HazeRadar API...")
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline(config)
        pipeline.initialize()
        
        # Load model
        logger.info("📦 Loading model...")
        model = RealtimeHazeGNN(
            in_feats=len(config.FEATURE_COLS),
            hidden=128,
            out_feats=1,
            num_heads=4,
            dropout=0.2
        ).to(device)
        
        if os.path.exists(config.MODEL_PATH):
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️ Model file not found, using untrained model")
        
        model.eval()
        
        # Initialize predictor
        predictor = PredictionEngine(model, pipeline, device)
        
        # Initial prediction
        update_predictions()
        
        # Schedule periodic updates
        scheduler.add_job(update_predictions, 'interval', seconds=config.UPDATE_INTERVAL)
        scheduler.start()
        
        logger.info("✅ API ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "last_update": predictor.last_update.isoformat() if predictor and predictor.last_update else None,
        "cities_count": len(pipeline.city_to_idx) if pipeline else 0,
        "version": "1.0.0"
    }

@app.get("/api/predictions/current", response_model=List[PredictionResponse])
async def get_current_predictions():
    """Get current haze predictions for all cities"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    
    try:
        # Return cached predictions if recent (< 5 minutes old)
        if (predictor.last_predictions and predictor.last_update and 
            (datetime.now() - predictor.last_update).seconds < 300):
            return predictor.last_predictions
        
        # Otherwise generate new predictions
        predictions = predictor.predict_current()
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/city/{city_name}", response_model=PredictionResponse)
async def get_city_prediction(city_name: str):
    """Get prediction for a specific city"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    
    try:
        predictions = predictor.last_predictions or predictor.predict_current()
        
        for pred in predictions:
            if pred['city'].lower() == city_name.lower():
                return pred
        
        raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"City prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/24h", response_model=List[ForecastResponse])
async def get_24h_forecast():
    """Get 24-hour haze forecast for all cities"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    
    try:
        current = predictor.last_predictions or predictor.predict_current()
        forecast = predictor.forecast_24h(current)
        return forecast
        
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/city/{city_name}", response_model=List[ForecastResponse])
async def get_city_forecast(city_name: str):
    """Get 24-hour forecast for a specific city"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    
    try:
        current = predictor.last_predictions or predictor.predict_current()
        forecast = predictor.forecast_24h(current)
        
        city_forecast = [f for f in forecast if f['city'].lower() == city_name.lower()]
        
        if not city_forecast:
            raise HTTPException(status_code=404, detail=f"City '{city_name}' not found")
        
        return city_forecast
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"City forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update")
async def trigger_update(background_tasks: BackgroundTasks):
    """Manually trigger prediction update"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")
    
    background_tasks.add_task(update_predictions)
    return {"message": "Update triggered"}

@app.get("/api/stats")
async def get_stats():
    """Get current statistics"""
    if not predictor or not predictor.last_predictions:
        raise HTTPException(status_code=503, detail="No predictions available")
    
    predictions = predictor.last_predictions
    
    pm25_values = [p['predicted_pm25'] for p in predictions]
    
    return {
        "total_cities": len(predictions),
        "avg_pm25": sum(pm25_values) / len(pm25_values),
        "max_pm25": max(pm25_values),
        "min_pm25": min(pm25_values),
        "high_risk_cities": len([p for p in predictions if p['predicted_pm25'] > 55.4]),
        "last_update": predictor.last_update.isoformat()
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
