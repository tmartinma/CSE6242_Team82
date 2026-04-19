import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Optional


# --- PRIORITY: ENVIRONMENT STABILITY PATCHES ---
# These patches resolve common Python 3.12 metadata and pycparser errors
# that can occur when loading AutoGluon models on modern macOS/Windows systems.
def apply_stability_patches():
    # 1. Mock pycparser if missing or broken (common on macOS/3.12)
    try:
        import pycparser

        if not hasattr(pycparser, "__version__"):
            pycparser.__version__ = "2.21"
    except Exception:
        pass

    # 2. Patch AutoGluon's version checker to prevent NoneType attribute errors
    def patched_get_package_versions():
        import importlib.metadata

        package_version_dict = {}
        for dist in importlib.metadata.distributions():
            try:
                name = dist.metadata.get("Name") or dist.metadata.get("name")
                if name:
                    package_version_dict[name.lower()] = dist.version
            except Exception:
                continue
        return package_version_dict

    try:
        import autogluon.common.utils.utils as ag_utils

        ag_utils.get_package_versions = patched_get_package_versions
        import autogluon.core.utils.utils as ag_core_utils

        ag_core_utils.get_package_versions = patched_get_package_versions
    except Exception:
        pass


# Apply patches before importing AutoGluon
apply_stability_patches()

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NYC Taxi Dual-ML API")

# Enable CORS for the Leaflet/React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path Configuration ---
# Update these paths if your folders are named differently (e.g., taxi_fare_quick vs taxi_fair_quick)
MODEL_PATH_DURATION = "AutogluonModels/taxi_duration_quick"
MODEL_PATH_FARE = "AutogluonModels/taxi_fare_quick"
CENTROID_CSV = "data/taxi_zone_centroids.csv"

# --- Global Predictors ---
_predictor_duration = None
_predictor_fare = None
_zones_df = None


def get_duration_predictor():
    global _predictor_duration
    if _predictor_duration is None:
        from autogluon.tabular import TabularPredictor

        logger.info(f"Loading Duration Model from {MODEL_PATH_DURATION}...")
        _predictor_duration = TabularPredictor.load(
            MODEL_PATH_DURATION,
            require_version_match=False,
            require_py_version_match=False,
            verbosity=0,
        )
    return _predictor_duration


def get_fare_predictor():
    global _predictor_fare
    if _predictor_fare is None:
        from autogluon.tabular import TabularPredictor

        logger.info(f"Loading Fare Model from {MODEL_PATH_FARE}...")
        _predictor_fare = TabularPredictor.load(
            MODEL_PATH_FARE,
            require_version_match=False,
            require_py_version_match=False,
            verbosity=0,
        )
    return _predictor_fare


def get_zones_df():
    global _zones_df
    if _zones_df is None:
        if Path(CENTROID_CSV).exists():
            _zones_df = pd.read_csv(CENTROID_CSV).set_index("LocationID")
        else:
            logger.warning(
                f"File {CENTROID_CSV} not found. Zone names will be unavailable."
            )
    return _zones_df


# --- Pydantic Models ---
class TripInput(BaseModel):
    trip_distance: float
    pu_location_id: int
    do_location_id: int


class PredictRequest(BaseModel):
    trips: List[TripInput]


# --- API Endpoints ---


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Simultaneously runs inference on the Duration and Fare models.
    """
    try:
        duration_engine = get_duration_predictor()
        fare_engine = get_fare_predictor()
        zones_data = get_zones_df()

        # Prepare data for AutoGluon models
        input_data = pd.DataFrame(
            [
                {
                    "PULocationID": t.pu_location_id,
                    "DOLocationID": t.do_location_id,
                    "trip_distance": t.trip_distance,
                }
                for t in req.trips
            ]
        )

        # Run dual inference
        duration_preds = duration_engine.predict(input_data)
        fare_preds = fare_engine.predict(input_data)

        results = []
        for i in range(len(req.trips)):
            pu_id = req.trips[i].pu_location_id
            do_id = req.trips[i].do_location_id

            # Resolve zone names for UI display
            pu_name, do_name = f"Zone {pu_id}", f"Zone {do_id}"
            if zones_data is not None:
                if pu_id in zones_data.index:
                    row = zones_data.loc[pu_id]
                    pu_name = f"{row['zone']} ({row['borough']})"
                if do_id in zones_data.index:
                    row = zones_data.loc[do_id]
                    do_name = f"{row['zone']} ({row['borough']})"

            results.append(
                {
                    "predicted_duration_minutes": round(
                        float(duration_preds.iloc[i]), 2
                    ),
                    "predicted_fare_amount": round(float(fare_preds.iloc[i]), 2),
                    "pickup_zone": pu_name,
                    "dropoff_zone": do_name,
                }
            )

        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Prediction logic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/zones/nearby")
def zones_nearby(lat: float, lon: float, top_n: int = 1):
    """
    Finds the nearest Taxi Zone ID based on map coordinates using the Haversine formula.
    """
    df = get_zones_df()
    if df is None:
        raise HTTPException(status_code=404, detail="Zone CSV data missing")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    temp_df = df.reset_index().copy()
    temp_df["distance_km"] = temp_df.apply(
        lambda r: haversine(lat, lon, r["centroid_lat"], r["centroid_lon"]), axis=1
    )
    return temp_df.nsmallest(top_n, "distance_km").to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "online", "models_ready": ["duration", "fare"]}


if __name__ == "__main__":
    import uvicorn

    # Start the server on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
