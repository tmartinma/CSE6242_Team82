import logging
from pathlib import Path
from typing import List

import pandas as pd
from autogluon.tabular import TabularPredictor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Taxi ML Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Ensure this matches your actual folder structure exactly
MODEL_PATH = "AutogluonModels/yellow_duration_quick"
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            logger.error(f"CRITICAL: Model directory NOT FOUND at {path.absolute()}")
            return None
        try:
            _predictor = TabularPredictor.load(str(path))
            logger.info("Predictor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            return None
    return _predictor


class TripInput(BaseModel):
    pickup_datetime: str
    trip_distance: float
    pu_location_id: int
    do_location_id: int


class PredictRequest(BaseModel):
    trips: List[TripInput]


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "status": "online",
        "model_loaded": predictor is not None,
        "absolute_path": str(Path(MODEL_PATH).absolute()),
    }


# --- NEW: Debug endpoint to inspect model features and importance ---
@app.get("/model-info")
def model_info():
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Please train the model first.",
        )
    try:
        importance = predictor.feature_importance(silent=True).to_dict()
        return {
            "features": str(predictor.feature_metadata_in),
            "feature_importance": importance,
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict")
async def predict(req: PredictRequest):
    predictor = get_predictor()
    if not predictor:
        # Provide a descriptive error instead of a 404
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Please train the model first.",
        )

    try:
        rows = []
        for t in req.trips:
            # --- FIX: Dynamically estimate fare based on trip distance ---
            # Instead of hardcoding fare_amount/total_amount (which caused
            # the model to always return the same prediction), we now
            # estimate them from trip_distance so inputs vary per trip.
            estimated_fare = round(t.trip_distance * 2.5 + 3.0, 2)   # ~$2.50/mile + $3 base
            estimated_total = round(estimated_fare + 3.5, 2)          # add tax/surcharge buffer
            rows.append(
                {
                    "VendorID": 1,
                    "tpep_pickup_datetime": pd.to_datetime(t.pickup_datetime),
                    "passenger_count": 1.0,
                    "trip_distance": t.trip_distance,
                    "RatecodeID": 1.0,
                    "PULocationID": t.pu_location_id,
                    "DOLocationID": t.do_location_id,
                    "payment_type": 1,
                    "fare_amount": estimated_fare,      # was hardcoded 10.0
                    "extra": 0.0,
                    "mta_tax": 0.5,
                    "tip_amount": 0.0,
                    "tolls_amount": 0.0,
                    "improvement_surcharge": 1.0,
                    "total_amount": estimated_total,    # was hardcoded 15.0
                    "congestion_surcharge": 2.5,
                    "Airport_fee": 0.0,
                    "cbd_congestion_fee": 0.0,
                }
            )

        df = pd.DataFrame(rows)

        # Log the input so you can debug what the model receives
        logger.info(f"Prediction input:\n{df[['trip_distance', 'PULocationID', 'DOLocationID', 'fare_amount', 'total_amount']].to_string()}")

        preds = predictor.predict(df).tolist()

        # Log the output
        logger.info(f"Predictions: {preds}")

        return {"predictions": preds}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
