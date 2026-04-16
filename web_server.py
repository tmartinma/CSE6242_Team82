import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# --- Configuration ---
ML_ENGINE_URL = "http://127.0.0.1:8000"


def get_nearest_location(lat, lng):
    """
    Uses the ML engine's /zones/nearby endpoint to find the nearest
    taxi zone LocationID and name based on lat/lng coordinates.
    This replaces the broken DB lookup since local_data.db is empty.
    """
    try:
        response = requests.get(
            f"{ML_ENGINE_URL}/zones/nearby",
            params={"lat": lat, "lon": lng, "top_n": 1},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data:
            zone = data[0]
            return zone["location_id"], zone["zone"]
        return 263, "Unknown"
    except Exception as e:
        print(f"Zone lookup error: {e}")
        return 263, "Unknown"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def proxy_predict():
    ui_data = request.json

    # 1. Spatial lookup via ML engine /zones/nearby
    pu_id, pu_name = get_nearest_location(ui_data["start_lat"], ui_data["start_lng"])
    do_id, do_name = get_nearest_location(ui_data["end_lat"],   ui_data["end_lng"])

    distance = float(ui_data.get("distance", 1.0))

    # 2. Estimate fare dynamically from distance (avoids hardcoded values
    #    that caused the model to always return the same prediction)
    estimated_fare  = round(distance * 2.5 + 3.0, 2)   # ~$2.50/mile + $3 base
    estimated_total = round(estimated_fare + 3.5, 2)    # add tax/surcharge buffer

    # 3. Build payload matching local_api.py TripInput schema exactly
    ml_payload = {
        "trips": [
            {
                "pickup_datetime":        "2026-03-21 12:00:00",
                "trip_distance":          distance,
                "pu_location_id":         int(pu_id),
                "do_location_id":         int(do_id),
                "fare_amount":            estimated_fare,   # required by local_api.py
                "total_amount":           estimated_total,  # required by local_api.py
                "vendor_id":              1,
                "passenger_count":        1.0,
                "ratecode_id":            1.0,
                "payment_type":           1,
                "extra":                  0.0,
                "mta_tax":                0.5,
                "tip_amount":             0.0,
                "tolls_amount":           0.0,
                "improvement_surcharge":  1.0,
                "congestion_surcharge":   2.5,
                "airport_fee":            0.0,
                "cbd_congestion_fee":     0.0,
            }
        ]
    }

    try:
        response = requests.post(
            f"{ML_ENGINE_URL}/predict", json=ml_payload, timeout=10
        )
        response.raise_for_status()
        prediction = response.json()

        # local_api.py returns: {"results": [{"predicted_duration_minutes": ...}], ...}
        eta = round(prediction["results"][0]["predicted_duration_minutes"], 2)

        return jsonify(
            {
                "eta":     eta,
                "pickup":  pu_name,
                "dropoff": do_name,
            }
        )
    except Exception as e:
        return jsonify({"error": f"ML Engine offline: {str(e)}"}), 503


if __name__ == "__main__":
    app.run(port=5001, debug=True)