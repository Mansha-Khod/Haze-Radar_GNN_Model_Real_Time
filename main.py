import os
import json
import torch
import pandas as pd
import numpy as np
import networkx as nx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from torch_geometric.data import Data
from realtime_haze_gnn import create_model  # ensure this file exists

load_dotenv()

print(">>> FastAPI import successful")

# FastAPI app
app = FastAPI()
print(">>> FastAPI instance created")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(">>> CORS configured")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# TEMPORARILY DISABLE MODEL LOADING
# MODEL_PATH = "realtime_haze_gnn.pt"
# model = torch.load(MODEL_PATH, map_location=device)
# model.eval()
print(">>> Model loading temporarily disabled for debugging")


# ---------------- Graph Loading ----------------

city_graphs = {}

def load_city_graphs():
    global city_graphs
    print(">>> Loading city graph cache...")

    try:
        with open("city_graph_cache.json", "r") as f:
            cache = json.load(f)
            city_graphs.update(cache)

        print(">>> City graph cache loaded successfully")

    except Exception as e:
        print(f">>> Error loading city graph cache: {e}")
        city_graphs.clear()


# ---------------- API ROUTES ----------------

@app.get("/")
async def root():
    return {"status": "running", "message": "Debug Mode Live"}


@app.get("/cities")
async def get_cities():
    return {"cities": list(city_graphs.keys())}


# ---------------- STARTUP EVENT ----------------

@app.on_event("startup")
async def startup_event():
    print(">>> Startup event triggered")

    # Load Graphs
    load_city_graphs()

    print(">>> Graphs loaded successfully")

    # TEMP remove scheduler for debugging
    # scheduler.start()
    print(">>> Scheduler disabled temporarily")

    print(">>> Startup finished")


# ----------------------------------------------------

