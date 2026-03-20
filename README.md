# CSE6242_Team82
Spring 2026 Gatech CSE6242 Group Project

Data Links:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page <br>
Prosessed data URL through shared drive in parquet format: <br>

## AutoGluon Installation: https://auto.gluon.ai/stable/install.html <br>
conda create -n autogluon-win python=3.10 -y<br>
conda activate autogluon-win<br>

pip install -U pip<br>
pip install -U setuptools wheel<br>
pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu<br>


# NYC Taxi AutoML API

FastAPI wrapper around your AutoGluon trip-duration model.

## Setup

```bash
# 1. Install dependencies (inside your autogluon_win conda env)
pip install fastapi "uvicorn[standard]"
# autogluon, pandas, numpy, pyarrow should already be installed

# 2. Put main.py in the same folder that contains AutogluonModels/
#    (i.e. your project root where you ran the notebook)

# 3. Start the server
uvicorn main:app --reload --port 8000
```

API docs auto-generated at → http://127.0.0.1:8000/docs

---

## Endpoints

### GET `/health`
Check if the server is up and whether a model is loaded.

---

### POST `/train`
Kick off AutoGluon training in the background.

```json
{
  "data_path": "data/concate_data/yellow_tripdata_2025_all.parquet",
  "sample_n": 200000,
  "time_limit": 300,
  "model_path": "AutogluonModels/yellow_duration_quick"
}
```

Poll `GET /train/status` to check progress (`idle` → `running` → `done` / `error`).

---

### POST `/predict`
Predict trip duration (minutes) for one or more trips.

```json
{
  "trips": [
    {
      "pickup_datetime": "2025-03-01 08:30:00",
      "vendor_id": 1,
      "passenger_count": 1.0,
      "trip_distance": 2.5,
      "ratecode_id": 1.0,
      "pu_location_id": 161,
      "do_location_id": 236,
      "payment_type": 1,
      "fare_amount": 12.5,
      "extra": 3.5,
      "mta_tax": 0.5,
      "tip_amount": 2.0,
      "tolls_amount": 0.0,
      "improvement_surcharge": 1.0,
      "total_amount": 19.5,
      "congestion_surcharge": 2.5,
      "airport_fee": 0.0,
      "cbd_congestion_fee": 0.0
    }
  ]
}
```

Response:
```json
{
  "predictions": [14.32],
  "count": 1,
  "unit": "minutes",
  "model_path": "AutogluonModels/yellow_duration_quick"
}
```

---

### GET `/model/info`
Returns leaderboard, best model name, eval metric, and feature importances.

---

## Calling from your UI (JavaScript)

```js
// Health check
const health = await fetch("http://localhost:8000/health").then(r => r.json());

// Predict
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    trips: [{
      pickup_datetime: "2025-03-01 08:30:00",
      trip_distance: 2.5,
      pu_location_id: 161,
      do_location_id: 236,
      fare_amount: 12.5,
      total_amount: 19.5,
      // other fields use defaults if omitted
    }]
  })
});
const { predictions } = await res.json();
console.log(`Predicted duration: ${predictions[0].toFixed(1)} minutes`);
```

---

## Notes
- The model is loaded **lazily** on first `/predict` or `/model/info` call.
- Training runs in a **background thread** so the API stays responsive.
- CORS is open (`*`) for local development — restrict in production.