"""
NYC Taxi AutoGluon API
Endpoints:
  POST /train          - Train the AutoGluon model
  GET  /train/status   - Poll training progress
  POST /predict        - Predict trip duration for one or more trips
  GET  /model/info     - Get model leaderboard & metrics
  GET  /health         - Health check
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
    description="AutoGluon-based trip duration prediction for NYC Yellow & Green Taxi",
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
MODEL_PATH = "AutogluonModels/yellow_duration_quick"
_predictor = None          # loaded lazily
_training_status = {
    "status": "idle",      # idle | running | done | error
    "message": "",
    "started_at": None,
    "finished_at": None,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
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
        logger.info("Loading predictor from %s ...", MODEL_PATH)
        _predictor = TabularPredictor.load(str(path))
        logger.info("Predictor loaded.")
    return _predictor


def build_feature_row(trip: "TripInput") -> dict:
    """
    Convert a TripInput into the feature dict AutoGluon expects.
    Uses sensible defaults for fields not provided by the UI
    (fare_amount, total_amount, etc.) so the UI only needs to
    send pickup_datetime, trip_distance, pu_location_id, do_location_id.
    """
    pickup_dt = pd.to_datetime(trip.pickup_datetime)

    # Estimate fare from distance if not provided (simple NYC rate: $3 base + $2.50/mile)
    estimated_fare = round(3.0 + (trip.trip_distance * 2.50), 2)
    fare = trip.fare_amount if trip.fare_amount is not None else estimated_fare
    total = trip.total_amount if trip.total_amount is not None else round(fare + 3.0, 2)

    return {
        "VendorID": trip.vendor_id,
        "tpep_pickup_datetime": pickup_dt,
        "passenger_count": trip.passenger_count,
        "trip_distance": trip.trip_distance,
        "RatecodeID": trip.ratecode_id,
        "PULocationID": trip.pu_location_id,
        "DOLocationID": trip.do_location_id,
        "payment_type": trip.payment_type,
        "fare_amount": fare,
        "extra": trip.extra,
        "mta_tax": trip.mta_tax,
        "tip_amount": trip.tip_amount,
        "tolls_amount": trip.tolls_amount,
        "improvement_surcharge": trip.improvement_surcharge,
        "total_amount": total,
        "congestion_surcharge": trip.congestion_surcharge,
        "Airport_fee": trip.airport_fee,
        "cbd_congestion_fee": trip.cbd_congestion_fee,
    }


def _detect_datetime_cols(df: pd.DataFrame):
    """
    Detect pickup/dropoff datetime column names for both yellow (tpep_)
    and green (lpep_) taxi data and return standardized column names.
    """
    pickup_col  = next((c for c in df.columns if "pickup_datetime"  in c), None)
    dropoff_col = next((c for c in df.columns if "dropoff_datetime" in c), None)
    return pickup_col, dropoff_col


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class TripInput(BaseModel):
    """
    Single trip feature row.
    Only pickup_datetime, trip_distance, pu_location_id, do_location_id
    are required — everything else has sensible defaults so the UI
    doesn't need to supply financial fields.
    """
    pickup_datetime:        str   = Field(...,  example="2025-03-01 08:30:00")
    vendor_id:              int   = Field(1,    example=1)
    passenger_count:        float = Field(1.0,  example=1.0)
    trip_distance:          float = Field(...,  example=2.5)
    ratecode_id:            float = Field(1.0,  example=1.0)
    pu_location_id:         int   = Field(...,  example=161)
    do_location_id:         int   = Field(...,  example=236)
    payment_type:           int   = Field(1,    example=1)
    fare_amount:            Optional[float] = Field(None, example=12.5)
    extra:                  float = Field(0.0,  example=0.0)
    mta_tax:                float = Field(0.5,  example=0.5)
    tip_amount:             float = Field(0.0,  example=0.0)
    tolls_amount:           float = Field(0.0,  example=0.0)
    improvement_surcharge:  float = Field(1.0,  example=1.0)
    total_amount:           Optional[float] = Field(None, example=19.5)
    congestion_surcharge:   float = Field(2.5,  example=2.5)
    airport_fee:            float = Field(0.0,  example=0.0)
    cbd_congestion_fee:     float = Field(0.0,  example=0.0)


class PredictRequest(BaseModel):
    trips: List[TripInput]


class PredictResponse(BaseModel):
    predictions: List[float]   # predicted duration in minutes
    count: int
    unit: str = "minutes"
    model_path: str


class TrainRequest(BaseModel):
    # Supports both a single concatenated parquet OR a folder of parquet files
    data_path: str = Field(
        "data/concate_data/yellow_tripdata_2025_all.parquet",
        description=(
            "Path to a parquet file OR a folder of parquet files. "
            "Supports both yellow (tpep_) and green (lpep_) taxi data."
        ),
    )
    sample_n:   int = Field(200_000, ge=1000,  le=5_000_000)
    time_limit: int = Field(300,     ge=60,    description="AutoGluon time limit in seconds")
    model_path: str = Field(MODEL_PATH,        description="Where to save the trained model")


# ─── Background training task ─────────────────────────────────────────────────
def _train_task(req: TrainRequest):
    global _predictor, _training_status, MODEL_PATH
    from autogluon.tabular import TabularPredictor

    _training_status["status"]     = "running"
    _training_status["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _training_status["message"]    = ""

    try:
        data_path = Path(req.data_path)

        # ── Load data: single file OR folder of parquet files ──────────────────
        if data_path.is_dir():
            files = list(data_path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in folder: {data_path}")
            logger.info("Loading %d parquet files from %s ...", len(files), data_path)
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        elif data_path.is_file():
            logger.info("Loading parquet file: %s ...", data_path)
            df = pd.read_parquet(data_path)
        else:
            raise FileNotFoundError(
                f"data_path '{req.data_path}' is neither a file nor a folder. "
                "Check the path and try again."
            )

        logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

        # ── Feature engineering: works for both yellow (tpep_) and green (lpep_) ──
        pickup_col, dropoff_col = _detect_datetime_cols(df)
        if not pickup_col or not dropoff_col:
            raise ValueError(
                "Could not detect pickup/dropoff datetime columns. "
                f"Available columns: {list(df.columns)}"
            )

        df[pickup_col]  = pd.to_datetime(df[pickup_col])
        df[dropoff_col] = pd.to_datetime(df[dropoff_col])

        df["duration"] = (
            (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
        ).round(2)

        # Filter out bad trips (negative, zero, or unrealistically long)
        df = df[(df["duration"] > 0) & (df["duration"] <= 120)].copy()

        # Standardize datetime column name so AutoGluon sees consistent features
        if pickup_col != "tpep_pickup_datetime":
            df["tpep_pickup_datetime"] = df[pickup_col]

        # Drop columns not useful for prediction
        drop_cols = [
            c for c in [
                dropoff_col,            # leaks the label
                pickup_col,             # replaced by tpep_pickup_datetime
                "store_and_fwd_flag",   # not predictive
            ]
            if c in df.columns and c != "tpep_pickup_datetime"
        ]
        df = df.drop(columns=drop_cols)

        # Sample for speed
        sample_n = min(req.sample_n, len(df))
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info("Training on %d rows ...", len(df))

        # ── Train AutoGluon ────────────────────────────────────────────────────
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

        _predictor  = predictor
        MODEL_PATH  = req.model_path
        _training_status["status"]      = "done"
        _training_status["message"]     = (
            f"Trained on {sample_n} rows from '{req.data_path}'. "
            f"Model saved to '{req.model_path}'."
        )
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Training complete.")

    except Exception as exc:
        _training_status["status"]      = "error"
        _training_status["message"]     = str(exc)
        _training_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.exception("Training failed: %s", exc)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick health check — shows whether the model is loaded."""
    model_loaded  = _predictor is not None
    model_exists  = Path(MODEL_PATH).exists()
    return {
        "status":               "ok",
        "model_loaded":         model_loaded,
        "model_exists_on_disk": model_exists,
        "model_path":           MODEL_PATH,
        "training_status":      _training_status["status"],
    }


@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start AutoGluon training in the background.
    Supports yellow taxi (tpep_) and green taxi (lpep_) parquet files,
    either as a single file or a folder of files.
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

    Minimal payload example (other fields use defaults):
    {
      "trips": [{
        "pickup_datetime": "2025-03-01 08:30:00",
        "trip_distance": 2.5,
        "pu_location_id": 161,
        "do_location_id": 236
      }]
    }
    """
    predictor = get_predictor()
    rows  = [build_feature_row(t) for t in req.trips]
    df    = pd.DataFrame(rows)
    preds = predictor.predict(df).tolist()
    return PredictResponse(
        predictions=preds,
        count=len(preds),
        model_path=MODEL_PATH,
    )


@app.get("/model/info")
def model_info():
    """Return leaderboard and basic model metadata."""
    predictor  = get_predictor()
    leaderboard = predictor.leaderboard(silent=True)

    feature_imp = []
    if hasattr(predictor, "feature_importance"):
        try:
            feature_imp = (
                predictor.feature_importance(silent=True)
                .reset_index()
                .rename(columns={"index": "feature"})
                .to_dict(orient="records")
            )
        except Exception:
            pass

    return {
        "model_path":        MODEL_PATH,
        "problem_type":      predictor.problem_type,
        "eval_metric":       predictor.eval_metric,
        "best_model":        predictor.get_model_best(),
        "leaderboard":       leaderboard[
            ["model", "score_val", "pred_time_val", "fit_time"]
        ].to_dict(orient="records"),
        "feature_importance": feature_imp,
    }
