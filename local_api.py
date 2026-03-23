"""
NYC Taxi AutoGluon API
Endpoints:
  POST /train        - Train the AutoGluon model
  POST /predict      - Predict trip duration for one or more trips
  GET  /model/info   - Get model leaderboard & metrics
  GET  /health       - Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Taxi AutoML API",
    description="AutoGluon-based trip duration prediction for NYC Yellow Taxi",
    version="1.0.0",
)

# Allow all origins for local UI dev — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ────────────────────────────────────────────────────────────
MODEL_PATH = "AutogluonModels/yellow_duration_quick"
_predictor = None          # loaded lazily
_training_status = {
    "status": "idle",      # idle | running | done | error
    "message": "",
    "started_at": None,
    "finished_at": None,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
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
        "VendorID": trip.vendor_id,
        "tpep_pickup_datetime": pickup_dt,
        "passenger_count": trip.passenger_count,
        "trip_distance": trip.trip_distance,
        "RatecodeID": trip.ratecode_id,
        "PULocationID": trip.pu_location_id,
        "DOLocationID": trip.do_location_id,
        "payment_type": trip.payment_type,
        "fare_amount": trip.fare_amount,
        "extra": trip.extra,
        "mta_tax": trip.mta_tax,
        "tip_amount": trip.tip_amount,
        "tolls_amount": trip.tolls_amount,
        "improvement_surcharge": trip.improvement_surcharge,
        "total_amount": trip.total_amount,
        "congestion_surcharge": trip.congestion_surcharge,
        "Airport_fee": trip.airport_fee,
        "cbd_congestion_fee": trip.cbd_congestion_fee,
    }


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class TripInput(BaseModel):
    """Single trip feature row (mirrors yellow_model_df minus dropoff_datetime)."""
    pickup_datetime: str = Field(..., example="2025-03-01 08:30:00")
    vendor_id: int = Field(1, example=1)
    passenger_count: float = Field(1.0, example=1.0)
    trip_distance: float = Field(..., example=2.5)
    ratecode_id: float = Field(1.0, example=1.0)
    pu_location_id: int = Field(..., example=161)
    do_location_id: int = Field(..., example=236)
    payment_type: int = Field(1, example=1)
    fare_amount: float = Field(..., example=12.5)
    extra: float = Field(0.0, example=3.5)
    mta_tax: float = Field(0.5, example=0.5)
    tip_amount: float = Field(0.0, example=2.0)
    tolls_amount: float = Field(0.0, example=0.0)
    improvement_surcharge: float = Field(1.0, example=1.0)
    total_amount: float = Field(..., example=19.5)
    congestion_surcharge: float = Field(2.5, example=2.5)
    airport_fee: float = Field(0.0, example=0.0)
    cbd_congestion_fee: float = Field(0.0, example=0.0)


class PredictRequest(BaseModel):
    trips: List[TripInput]


class PredictResponse(BaseModel):
    predictions: List[float]            # predicted duration in minutes
    count: int
    unit: str = "minutes"
    model_path: str


class TrainRequest(BaseModel):
    data_path: str = Field(
        "data/concate_data/yellow_tripdata_2025_all.parquet",
        description="Path to yellow parquet file (relative to working dir)",
    )
    sample_n: int = Field(200_000, ge=1000, le=5_000_000)
    time_limit: int = Field(300, ge=60, description="AutoGluon time limit (seconds)")
    model_path: str = Field(MODEL_PATH, description="Where to save the model")


# ─── Background training task ─────────────────────────────────────────────────
def _train_task(req: TrainRequest):
    global _predictor, _training_status, MODEL_PATH
    from autogluon.tabular import TabularPredictor

    _training_status["status"] = "running"
    _training_status["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        logger.info("Loading data from %s …", req.data_path)
        df = pd.read_parquet(req.data_path)

        # Feature engineering (same as notebook)
        df["duration"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        ).round(2)
        df = df[(df["duration"] > 0) & (df["duration"] <= 70)].copy()

        drop_cols = [c for c in ["tpep_dropoff_datetime", "store_and_fwd_flag"] if c in df.columns]
        df = df.drop(columns=drop_cols)

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

        _predictor = predictor
        MODEL_PATH = req.model_path
        _training_status["status"] = "done"
        _training_status["message"] = f"Trained on {sample_n} rows. Model saved to {req.model_path}."
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Training complete.")
    except Exception as exc:
        _training_status["status"] = "error"
        _training_status["message"] = str(exc)
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.exception("Training failed: %s", exc)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    model_loaded = _predictor is not None
    model_exists = Path(MODEL_PATH).exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_exists_on_disk": model_exists,
        "model_path": MODEL_PATH,
    }


@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start AutoGluon training in the background.
    Poll GET /train/status to check progress.
    """
    if _training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    background_tasks.add_task(_train_task, req)
    return {"message": "Training started in background.", "config": req.dict()}


@app.get("/train/status")
def train_status():
    """Check the current training job status."""
    return _training_status


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict trip duration (minutes) for one or more trips.

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
    df = pd.DataFrame(rows)
    preds = predictor.predict(df).tolist()
    return PredictResponse(
        predictions=preds,
        count=len(preds),
        model_path=MODEL_PATH,
    )


@app.get("/model/info")
def model_info():
    """Return leaderboard and basic model metadata."""
    predictor = get_predictor()
    leaderboard = predictor.leaderboard(silent=True)
    return {
        "model_path": MODEL_PATH,
        "problem_type": predictor.problem_type,
        "eval_metric": predictor.eval_metric,
        "best_model": predictor.get_model_best(),
        "leaderboard": leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].to_dict(orient="records"),
        "feature_importance": predictor.feature_importance(silent=True).reset_index()
            .rename(columns={"index": "feature"})
            .to_dict(orient="records") if hasattr(predictor, "feature_importance") else [],
    }
