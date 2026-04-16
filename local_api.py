"""
NYC Taxi AutoGluon API
Endpoints:
  POST /train              - Train the AutoGluon model
  POST /predict            - Predict trip duration; response includes pickup/dropoff centroid coords
  GET  /model/info         - Get model leaderboard & metrics
  GET  /health             - Health check
  GET  /zones              - List all taxi zones with centroid coordinates
  GET  /zones/{location_id}- Look up a single zone by LocationID
  GET  /zones/nearby       - Find the nearest zone to a given lat/lon
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Taxi AutoML API",
    description="AutoGluon-based trip duration prediction for NYC Yellow Taxi",
    version="1.1.0",
)

# Allow all origins for local UI dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ─────────────────────────────────────────────────────────────
MODEL_PATH   = "AutogluonModels/taxi_duration_quick"
CENTROID_CSV = "data/taxi_zone_centroids.csv"

_predictor       = None   
_zones_df        = None   
_training_status = {
    "status":      "idle",
    "message":     "",
    "started_at":  None,
    "finished_at": None,
}


# ─── Zone helpers ──────────────────────────────────────────────────────────────
def get_zones_df() -> pd.DataFrame:
    global _zones_df
    if _zones_df is None:
        path = Path(CENTROID_CSV)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Centroid CSV not found at '{CENTROID_CSV}'.",
            )
        _zones_df = pd.read_csv(path).set_index("LocationID")
    return _zones_df


def zone_record(location_id: int) -> Optional[dict]:
    df = get_zones_df()
    if location_id not in df.index:
        return None
    row = df.loc[location_id]
    return {
        "location_id":   int(location_id),
        "zone":          row["zone"],
        "borough":       row["borough"],
        "centroid_lon":  float(row["centroid_lon"]),
        "centroid_lat":  float(row["centroid_lat"]),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── AutoGluon helpers ────────────────────────────────────────────────────────
def get_predictor():
    global _predictor
    if _predictor is None:
        from autogluon.tabular import TabularPredictor
        path = Path(MODEL_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found at '{MODEL_PATH}'. Call POST /train first.",
            )
        _predictor = TabularPredictor.load(str(path), require_version_match=False)
    return _predictor

def build_feature_row(trip: "TripInput") -> dict:
    """
    Matches the 4 features used in AutoML.ipynb exactly.
    Using VendorID or total_amount here would cause a model mismatch error.
    """
    return {
        "fare_amount":   trip.fare_amount,
        "PULocationID":  trip.pu_location_id,
        "DOLocationID":  trip.do_location_id,
        "trip_distance": trip.trip_distance,
    }


# ─── Pydantic schemas ──────────────────────────────────────────────────────────
class TripInput(BaseModel):
    pickup_datetime:       str
    trip_distance:         float
    pu_location_id:        int
    do_location_id:        int
    fare_amount:           float
    # These fields are included for compatibility with the Web Server proxy
    vendor_id:             Optional[int]   = 1
    passenger_count:       Optional[float] = 1.0
    total_amount:          Optional[float] = None
    congestion_surcharge:  Optional[float] = 2.5


class ZoneInfo(BaseModel):
    location_id:  int
    zone:         str
    borough:      str
    centroid_lon: float
    centroid_lat: float


class TripPrediction(BaseModel):
    predicted_duration_minutes: float
    pickup_zone:                Optional[ZoneInfo]
    dropoff_zone:               Optional[ZoneInfo]


class PredictRequest(BaseModel):
    trips: List[TripInput]


class PredictResponse(BaseModel):
    results:    List[TripPrediction]
    count:      int
    unit:       str = "minutes"
    model_path: str


class TrainRequest(BaseModel):
    data_path: str = "data/concate_data/combined_tripdata_2025_all.parquet"
    sample_n:   int = Field(100_000, ge=1000)
    time_limit: int = Field(300, ge=60)
    model_path: str = MODEL_PATH


class NearbyZone(BaseModel):
    location_id:    int
    zone:           str
    borough:        str
    centroid_lon:   float
    centroid_lat:   float
    distance_km:    float


# ─── Background training task ──────────────────────────────────────────────────
def _train_task(req: TrainRequest):
    global _predictor, _training_status, MODEL_PATH
    from autogluon.tabular import TabularPredictor

    _training_status["status"]     = "running"
    _training_status["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    try:
        logger.info("Loading data from %s …", req.data_path)
        df = pd.read_parquet(req.data_path)

        # 1. Resolve Column Names (Fixes KeyError)
        p_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
        d_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

        # 2. Compute Target Variable
        df["duration"] = (
            (df[d_col] - df[p_col]).dt.total_seconds() / 60
        ).round(2)

        # 3. Filter and Select Features (Matches Prediction Schema)
        df = df[(df["duration"] > 0) & (df["duration"] <= 70)].copy()
        feature_cols = ["fare_amount", "PULocationID", "DOLocationID", "trip_distance", "duration"]
        df = df[feature_cols].dropna().copy()

        # 4. Sampling
        sample_n = min(req.sample_n, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info("Training on %d rows with features: %s", len(df), feature_cols[:-1])

        predictor = TabularPredictor(
            label="duration",
            problem_type="regression",
            eval_metric="mae",
            path=req.model_path,
        )
        
        predictor.fit(
            train_data=df,
            presets=["medium_quality_faster_train", "optimize_for_deployment"],
            time_limit=req.time_limit,
            verbosity=2,
        )

        _predictor                      = predictor
        MODEL_PATH                      = req.model_path
        _training_status["status"]      = "done"
        _training_status["message"]     = f"Success! Model saved to {req.model_path}."
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    except Exception as exc:
        _training_status["status"]      = "error"
        _training_status["message"]     = str(exc)
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.exception("Training failed.")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "model_path": MODEL_PATH
    }

@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if _training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training in progress.")
    background_tasks.add_task(_train_task, req)
    return {"message": "Training started."}

@app.get("/train/status")
def train_status():
    return _training_status

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = get_predictor()
    
    # Strip incoming JSON to only the 4 features the model knows
    rows = [build_feature_row(t) for t in req.trips]
    df = pd.DataFrame(rows)
    preds = predictor.predict(df).tolist()

    try:
        get_zones_df()
        zones_available = True
    except:
        zones_available = False

    results = []
    for trip, duration in zip(req.trips, preds):
        p_zone = zone_record(trip.pu_location_id) if zones_available else None
        d_zone = zone_record(trip.do_location_id) if zones_available else None
        results.append(TripPrediction(
            predicted_duration_minutes=max(0, round(duration, 2)),
            pickup_zone=ZoneInfo(**p_zone) if p_zone else None,
            dropoff_zone=ZoneInfo(**d_zone) if d_zone else None,
        ))

    return PredictResponse(results=results, count=len(results), model_path=MODEL_PATH)

@app.get("/zones", response_model=List[ZoneInfo])
def list_zones(borough: Optional[str] = None):
    df = get_zones_df().reset_index()
    if borough:
        df = df[df["borough"].str.lower() == borough.lower()]
    return [ZoneInfo(**row.to_dict()) for _, row in df.iterrows()]

@app.get("/zones/nearby", response_model=List[NearbyZone])
def zones_nearby(lat: float, lon: float, top_n: int = 5):
    df = get_zones_df().reset_index().copy()
    df["distance_km"] = df.apply(lambda r: haversine_km(lat, lon, r["centroid_lat"], r["centroid_lon"]), axis=1)
    nearest = df.nsmallest(top_n, "distance_km")
    return [NearbyZone(**row.to_dict()) for _, row in nearest.iterrows()]

@app.get("/zones/{location_id}", response_model=ZoneInfo)
def get_zone(location_id: int):
    record = zone_record(location_id)
    if not record: raise HTTPException(status_code=404)
    return ZoneInfo(**record)