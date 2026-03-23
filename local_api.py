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

# Allow all origins for local UI dev — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ─────────────────────────────────────────────────────────────
MODEL_PATH   = "AutogluonModels/yellow_duration_quick"
CENTROID_CSV = "data/taxi_zone_centroids.csv"   # path relative to working dir

_predictor       = None   # AutoGluon predictor, loaded lazily
_zones_df        = None   # centroid lookup table, loaded lazily
_training_status = {
    "status":      "idle",  # idle | running | done | error
    "message":     "",
    "started_at":  None,
    "finished_at": None,
}


# ─── Zone helpers ──────────────────────────────────────────────────────────────
def get_zones_df() -> pd.DataFrame:
    """
    Load (or return cached) the centroid CSV as a DataFrame indexed by LocationID.
    Expected columns: LocationID, zone, borough, centroid_lon, centroid_lat
    """
    global _zones_df
    if _zones_df is None:
        path = Path(CENTROID_CSV)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Centroid CSV not found at '{CENTROID_CSV}'. "
                       "Generate it first with the geo-processing script.",
            )
        logger.info("Loading centroid CSV from %s …", CENTROID_CSV)
        _zones_df = pd.read_csv(path).set_index("LocationID")
        logger.info("Loaded %d zones.", len(_zones_df))
    return _zones_df


def zone_record(location_id: int) -> Optional[dict]:
    """Return a single zone dict for location_id, or None if not found."""
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
    """Great-circle distance in km between two (lat, lon) points."""
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
    """Load (or return cached) AutoGluon predictor."""
    global _predictor
    if _predictor is None:
        from autogluon.tabular import TabularPredictor
        path = Path(MODEL_PATH)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found at '{MODEL_PATH}'. Call POST /train first.",
            )
        logger.info("Loading predictor from %s …", MODEL_PATH)
        _predictor = TabularPredictor.load(str(path))
        logger.info("Predictor loaded.")
    return _predictor


def build_feature_row(trip: "TripInput") -> dict:
    """Convert a TripInput into the feature dict AutoGluon expects."""
    pickup_dt = pd.to_datetime(trip.pickup_datetime)
    return {
        "VendorID":             trip.vendor_id,
        "tpep_pickup_datetime": pickup_dt,
        "passenger_count":      trip.passenger_count,
        "trip_distance":        trip.trip_distance,
        "RatecodeID":           trip.ratecode_id,
        "PULocationID":         trip.pu_location_id,
        "DOLocationID":         trip.do_location_id,
        "payment_type":         trip.payment_type,
        "fare_amount":          trip.fare_amount,
        "extra":                trip.extra,
        "mta_tax":              trip.mta_tax,
        "tip_amount":           trip.tip_amount,
        "tolls_amount":         trip.tolls_amount,
        "improvement_surcharge":trip.improvement_surcharge,
        "total_amount":         trip.total_amount,
        "congestion_surcharge": trip.congestion_surcharge,
        "Airport_fee":          trip.airport_fee,
        "cbd_congestion_fee":   trip.cbd_congestion_fee,
    }


# ─── Pydantic schemas ──────────────────────────────────────────────────────────
class TripInput(BaseModel):
    """Single trip feature row (mirrors yellow_model_df minus dropoff_datetime)."""
    pickup_datetime:       str   = Field(...,  example="2025-03-01 08:30:00")
    vendor_id:             int   = Field(1,    example=1)
    passenger_count:       float = Field(1.0,  example=1.0)
    trip_distance:         float = Field(...,  example=2.5)
    ratecode_id:           float = Field(1.0,  example=1.0)
    pu_location_id:        int   = Field(...,  example=161)
    do_location_id:        int   = Field(...,  example=236)
    payment_type:          int   = Field(1,    example=1)
    fare_amount:           float = Field(...,  example=12.5)
    extra:                 float = Field(0.0,  example=3.5)
    mta_tax:               float = Field(0.5,  example=0.5)
    tip_amount:            float = Field(0.0,  example=2.0)
    tolls_amount:          float = Field(0.0,  example=0.0)
    improvement_surcharge: float = Field(1.0,  example=1.0)
    total_amount:          float = Field(...,  example=19.5)
    congestion_surcharge:  float = Field(2.5,  example=2.5)
    airport_fee:           float = Field(0.0,  example=0.0)
    cbd_congestion_fee:    float = Field(0.0,  example=0.0)


class ZoneInfo(BaseModel):
    """Centroid metadata for one taxi zone."""
    location_id:  int
    zone:         str
    borough:      str
    centroid_lon: float
    centroid_lat: float


class TripPrediction(BaseModel):
    """Prediction result for a single trip, enriched with zone coordinates."""
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
    data_path: str = Field(
        "data/concate_data/yellow_tripdata_2025_all.parquet",
        description="Path to yellow parquet file (relative to working dir)",
    )
    sample_n:   int = Field(200_000, ge=1000,  le=5_000_000)
    time_limit: int = Field(300,     ge=60,    description="AutoGluon time limit (seconds)")
    model_path: str = Field(MODEL_PATH,        description="Where to save the model")


class NearbyZone(BaseModel):
    """Nearest zone result, includes distance from the query point."""
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

        # Feature engineering — compute trip duration in minutes
        df["duration"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        ).round(2)

        # Keep only valid durations (0–70 min), matching notebook filter
        df = df[(df["duration"] > 0) & (df["duration"] <= 70)].copy()

        # Drop columns that leak the target or are always null
        drop_cols = [c for c in ["tpep_dropoff_datetime", "store_and_fwd_flag"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Sub-sample for faster training runs
        sample_n = min(req.sample_n, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info("Training on %d rows …", len(df))

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
        _training_status["message"]     = (
            f"Trained on {sample_n} rows. Model saved to {req.model_path}."
        )
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Training complete.")

    except Exception as exc:
        _training_status["status"]      = "error"
        _training_status["message"]     = str(exc)
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.exception("Training failed: %s", exc)


# ─── Routes — health & training ───────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    """Return server status, model load state, and centroid CSV availability."""
    model_loaded   = _predictor is not None
    model_exists   = Path(MODEL_PATH).exists()
    centroid_ready = _zones_df is not None or Path(CENTROID_CSV).exists()
    return {
        "status":             "ok",
        "model_loaded":       model_loaded,
        "model_exists_on_disk": model_exists,
        "model_path":         MODEL_PATH,
        "centroid_csv_ready": centroid_ready,
        "centroid_csv_path":  CENTROID_CSV,
    }


@app.post("/train", summary="Start model training")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start AutoGluon training in the background.
    Poll GET /train/status to check progress.
    """
    if _training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    background_tasks.add_task(_train_task, req)
    return {"message": "Training started in background.", "config": req.dict()}


@app.get("/train/status", summary="Check training job status")
def train_status():
    """Returns current training status: idle | running | done | error."""
    return _training_status


# ─── Routes — prediction ──────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, summary="Predict trip duration")
def predict(req: PredictRequest):
    """
    Predict trip duration (minutes) for one or more trips.
    Each result is enriched with the pickup and dropoff zone centroid
    coordinates looked up from the centroid CSV.

    Example single-trip payload:
    {
      "trips": [{
        "pickup_datetime": "2025-03-01 08:30:00",
        "trip_distance": 2.5,
        "pu_location_id": 161,
        "do_location_id": 236,
        "fare_amount": 12.5,
        "total_amount": 19.5
      }]
    }
    """
    predictor = get_predictor()

    rows = [build_feature_row(t) for t in req.trips]
    df   = pd.DataFrame(rows)
    preds = predictor.predict(df).tolist()

    # Attempt to load zone info — if CSV is missing, zones will be None
    try:
        get_zones_df()
        zones_available = True
    except HTTPException:
        zones_available = False
        logger.warning("Centroid CSV not available — zone info omitted from response.")

    results = []
    for trip, duration in zip(req.trips, preds):
        pickup_zone  = zone_record(trip.pu_location_id)  if zones_available else None
        dropoff_zone = zone_record(trip.do_location_id)  if zones_available else None
        results.append(TripPrediction(
            predicted_duration_minutes=round(duration, 2),
            pickup_zone=ZoneInfo(**pickup_zone)   if pickup_zone  else None,
            dropoff_zone=ZoneInfo(**dropoff_zone) if dropoff_zone else None,
        ))

    return PredictResponse(
        results=results,
        count=len(results),
        model_path=MODEL_PATH,
    )


# ─── Routes — zones ───────────────────────────────────────────────────────────
@app.get("/zones", response_model=List[ZoneInfo], summary="List all taxi zones")
def list_zones(borough: Optional[str] = Query(None, description="Filter by borough name")):
    """
    Return all 263 NYC taxi zones with their centroid coordinates.
    Optionally filter by borough (e.g. Manhattan, Brooklyn, Queens, Bronx, Staten Island, EWR).
    """
    df = get_zones_df().reset_index()
    if borough:
        df = df[df["borough"].str.lower() == borough.lower()]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No zones found for borough '{borough}'.",
            )
    return [
        ZoneInfo(
            location_id=int(row["LocationID"]),
            zone=row["zone"],
            borough=row["borough"],
            centroid_lon=float(row["centroid_lon"]),
            centroid_lat=float(row["centroid_lat"]),
        )
        for _, row in df.iterrows()
    ]


@app.get("/zones/nearby", response_model=List[NearbyZone], summary="Find nearest zones to a coordinate")
def zones_nearby(
    lat:     float = Query(..., description="Latitude of query point",  example=40.7580),
    lon:     float = Query(..., description="Longitude of query point", example=-73.9855),
    top_n:   int   = Query(5,   ge=1, le=50, description="Number of nearest zones to return"),
):
    """
    Return the top_n taxi zones nearest to the given (lat, lon) coordinate,
    sorted by great-circle distance (km).

    Useful for matching a user's GPS location to a LocationID before calling /predict.
    """
    df = get_zones_df().reset_index()

    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: haversine_km(lat, lon, r["centroid_lat"], r["centroid_lon"]),
        axis=1,
    )
    nearest = df.nsmallest(top_n, "distance_km")

    return [
        NearbyZone(
            location_id=int(row["LocationID"]),
            zone=row["zone"],
            borough=row["borough"],
            centroid_lon=float(row["centroid_lon"]),
            centroid_lat=float(row["centroid_lat"]),
            distance_km=round(float(row["distance_km"]), 4),
        )
        for _, row in nearest.iterrows()
    ]


@app.get("/zones/{location_id}", response_model=ZoneInfo, summary="Look up a zone by LocationID")
def get_zone(location_id: int):
    """
    Return zone name, borough, and centroid coordinates for a single LocationID.
    LocationID matches the PULocationID / DOLocationID columns in the taxi trip data.
    """
    record = zone_record(location_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"LocationID {location_id} not found in centroid CSV.",
        )
    return ZoneInfo(**record)