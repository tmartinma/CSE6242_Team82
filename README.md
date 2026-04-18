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
## Running the Server

### Step 1 — Project structure
Make sure `main.py` is in the same folder as your `AutogluonModels/` directory:
```
project-root/
├── main.py
├── AutogluonModels/
│   └── yellow_duration_quick/
└── data/
    └── concate_data/
        └── yellow_tripdata_2025_all.parquet
```

### Step 2 — Start the server
```bash
conda activate autogluon-win
uvicorn main:app --reload --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### Step 3 — Verify the server is running
Open your browser and go to:
- **Interactive API docs (Swagger UI):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

Expected health response:
```json
{ "status": "ok", "model_loaded": false }
```
---

## Endpoints

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if server and model are running |
| POST | `/train` | Kick off AutoGluon training in the background |
| GET | `/train/status` | Poll training progress (`idle` → `running` → `done` / `error`) |
| POST | `/predict` | Predict trip duration (in minutes) for one or more trips |
| GET | `/model/info` | Returns leaderboard, best model, eval metric, and feature importances |

---

## Testing the API

### Option A — Swagger UI (Recommended for quick testing)
1. Navigate to http://127.0.0.1:8000/docs
2. Click any endpoint → **"Try it out"** → fill in parameters → **"Execute"**

### Option B — cURL

**Health check:**
```bash
curl http://localhost:8000/health
```

**Trigger training (Yellow Taxi):**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "taxi_type": "yellow",
    "data_path": "data/concate_data/yellow_tripdata_2025_all.parquet",
    "sample_n": 200000,
    "time_limit": 300,
    "model_path": "AutogluonModels/yellow_fare_model"
  }'
```

**Trigger training (Green Taxi):**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "taxi_type": "green",
    "data_path": "data/concate_data/green_tripdata_2025_all.parquet",
    "sample_n": 200000,
    "time_limit": 300,
    "model_path": "AutogluonModels/green_fare_model"
  }'
```

**Check training status:**
```bash
curl http://localhost:8000/train/status
```

**Predict distance and fare (Yellow Taxi):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "taxi_type": "yellow",
    "trips": [{
      "pickup_datetime": "2025-03-01 08:30:00",
      "vendor_id": 1,
      "passenger_count": 1.0,
      "ratecode_id": 1.0,
      "pu_location_id": 161,
      "do_location_id": 236,
      "payment_type": 1,
      "extra": 3.5,
      "mta_tax": 0.5,
      "tip_amount": 2.0,
      "tolls_amount": 0.0,
      "improvement_surcharge": 1.0,
      "congestion_surcharge": 2.5,
      "airport_fee": 0.0,
      "cbd_congestion_fee": 0.0
    }]
  }'
```

**Predict distance and fare (Green Taxi):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "taxi_type": "green",
    "trips": [{
      "pickup_datetime": "2025-03-01 08:30:00",
      "vendor_id": 2,
      "passenger_count": 1.0,
      "ratecode_id": 1.0,
      "pu_location_id": 74,
      "do_location_id": 75,
      "payment_type": 1,
      "extra": 0.5,
      "mta_tax": 0.5,
      "tip_amount": 1.5,
      "tolls_amount": 0.0,
      "improvement_surcharge": 1.0,
      "congestion_surcharge": 2.5
    }]
  }'
```

Expected prediction response:
```json
{
  "taxi_type": "yellow",
  "predictions": [
    {
      "trip_distance": 2.53,
      "fare_amount": 12.80
    }
  ],
  "count": 1,
  "units": {
    "trip_distance": "miles",
    "fare_amount": "USD"
  },
  "model_path": "AutogluonModels/yellow_fare_model"
}
```

### Option C — JavaScript (for UI integration)
```js
// Health check
const health = await fetch("http://localhost:8000/health").then(r => r.json());

// Predict distance and fare — set taxi_type to "yellow" or "green"
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    taxi_type: "yellow",
    trips: [{
      pickup_datetime: "2025-03-01 08:30:00",
      pu_location_id: 161,
      do_location_id: 236,
      passenger_count: 1.0,
      payment_type: 1
      // other fields will use defaults if omitted
    }]
  })
});
const { predictions } = await res.json();
console.log(`Predicted distance: ${predictions[0].trip_distance.toFixed(2)} miles`);
console.log(`Predicted fare: $${predictions[0].fare_amount.toFixed(2)}`);
```

---

## Notes
- The model is loaded **lazily** on first `/predict` or `/model/info` call.
- Training runs in a **background thread** so the API stays responsive.
- CORS is open (`*`) for local development — restrict in production.