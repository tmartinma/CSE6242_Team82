import os

import requests
from flask import Flask, jsonify, render_template, request
from sqlalchemy import create_engine, text

app = Flask(__name__)

# --- Configuration ---
DB_PATH = "local_data.db"
ML_ENGINE_URL = "http://127.0.0.1:8000"
# CRITICAL: Set this to the exact name of your table in the DB
MAIN_TABLE = "final_combined_data"
engine = create_engine(f"sqlite:///{DB_PATH}")


def get_nearest_location(lat, lng):
    """
    Queries the main table to find the nearest LocationID based on
    the coordinates stored in your data.
    """
    # We use PULocationID and zone from your main table columns
    query = text(
        f"""
        SELECT PULocationID, zone
        FROM {MAIN_TABLE}
        ORDER BY ((latitude - :lat)*(latitude - :lat) + (longitude - :lng)*(longitude - :lng)) ASC
        LIMIT 1
    """
    )
    try:
        with engine.connect() as conn:
            res = conn.execute(query, {"lat": lat, "lng": lng}).fetchone()
            if res:
                return res[0], res[1]  # Returns (LocationID, ZoneName)
            return 263, "Unknown"
    except Exception as e:
        print(f"Database Query Error: {e}")
        return 263, "Database Error"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def proxy_predict():
    ui_data = request.json

    # 1. Spatial Lookup using your combined data table
    pu_id, pu_name = get_nearest_location(ui_data["start_lat"], ui_data["start_lng"])
    do_id, do_name = get_nearest_location(ui_data["end_lat"], ui_data["end_lng"])

    # 2. Forward to ML Engine (FastAPI)
    ml_payload = {
        "trips": [
            {
                "pickup_datetime": "2026-03-21 12:00:00",
                "trip_distance": float(ui_data.get("distance", 1.0)),
                "pu_location_id": int(pu_id),
                "do_location_id": int(do_id),
            }
        ]
    }

    try:
        response = requests.post(f"{ML_ENGINE_URL}/predict", json=ml_payload, timeout=5)
        response.raise_for_status()
        prediction = response.json()

        return jsonify(
            {
                "eta": round(prediction["predictions"][0], 2),
                "pickup": pu_name,
                "dropoff": do_name,
            }
        )
    except Exception as e:
        return jsonify({"error": f"ML Engine offline: {str(e)}"}), 503


if __name__ == "__main__":
    app.run(port=5001, debug=True)
