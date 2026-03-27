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
                    "fare_amount": 10.0,
                    "extra": 0.0,
                    "mta_tax": 0.5,
                    "tip_amount": 0.0,
                    "tolls_amount": 0.0,
                    "improvement_surcharge": 1.0,
                    "total_amount": 15.0,
                    "congestion_surcharge": 2.5,
                    "Airport_fee": 0.0,
                    "cbd_congestion_fee": 0.0,
                }
            )

        df = pd.DataFrame(rows)
        preds = predictor.predict(df).tolist()
        return {"predictions": preds}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
