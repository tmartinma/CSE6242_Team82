# CSE6242_Team82
Spring 2026 Gatech CSE6242 Group Project
Spring 2026 | Georgia Tech CSE6242 Group Project

This project predicts **trip distance** and **fare price** for NYC **Yellow Taxi** and **Green Taxi** trips using AutoGluon AutoML, paired with an interactive map visualization dashboard.

---

## Background

Transportation is an important part of daily life for millions of people. In a dense, fast-paced city like New York City, getting around can be challenging. Public transportation and private cars share the same streets, which often leads to heavy traffic affecting residents, commuters, and visitors. Even though subways and ride-sharing services have grown in recent years, the city's well-known taxi fleet — including yellow and green cabs — remains a critical part of daily transportation, with thousands of taxis traveling through the five boroughs every day.

In a city where time matters, traffic can make a short trip take much longer than expected. This project addresses this problem by building a system that predicts taxi trip fare and distance, and visualizes mobility patterns across the city. By turning large amounts of taxi data into clear predictions and visual insights, the project helps people better understand travel patterns and supports more informed decisions for both city planners and passengers.

---

## Problem Statement

In New York City, taxi fare and trip distance can vary significantly depending on location, time, and traffic conditions. A trip that looks short on the map may still cost more or cover a different distance than expected due to congestion or differences between areas of the city. This makes it difficult for passengers to anticipate costs and distances, and also makes it harder to understand overall travel patterns from raw taxi records.

We define the problem as **predicting taxi trip distance and fare price** based on historical yellow and green taxi trip data. The model uses features such as pickup time, pickup location, and drop-off location to estimate these values. Beyond just outputting numbers, our goal is to combine prediction with an interactive map interface so users can visually explore trip estimates and better understand mobility patterns across New York City.

---

## Method

### Data
We use 2025 NYC TLC taxi trip data, aggregating 11 months of yellow and green taxi records — over **44 million trips** in total. We perform geospatial mapping using GeoPandas and TLC taxi zone files to convert pickup and dropoff LocationIDs into latitude/longitude centroids. Extreme outliers are removed by keeping only trips with durations between 0 and 70 minutes. A memory-optimized SQLite pipeline is used to handle large-scale data joins without system failures.

### AutoML Modeling
The latest `AutoML.ipynb` uses a compact, deployment-focused AutoGluon setup rather than a broad multi-model search. It standardizes the taxi datetime columns to `pickup_datetime` and `dropoff_datetime` for both yellow and green data, then trains two lightweight ensembles: `taxi_duration_quick` and `taxi_fare_quick`. The notebook uses only LightGBM and CatBoost, with `optimize_for_deployment` enabled and no bagging or stacking, so the saved models stay small and fast to load. On the sampled holdout set, this setup reaches roughly **4.03 MAE** for duration and **4.89 MAE** for fare.

AutoGluon is a good fit here because it reduces manual model tuning while still producing strong results. The built-in ensemble structure helps combine the strengths of the underlying models, which improves prediction quality compared with relying on a single hand-picked model. In practice, this also makes the training workflow faster to iterate on and easier to deploy locally, since the final models are compact and ready to load without extra tuning.

### System Architecture
The system is built in three layers:
1. **Data pipeline** — memory-efficient SQLite setup with geospatial mapping via GeoPandas
2. **Modeling** — AutoGluon AutoML with automatic feature selection and model ensembling
3. **Backend API** — FastAPI serving real-time predictions with nearest taxi zone lookups, connected to an interactive frontend dashboard ("TaxiPredict.ai")

### Visualization
An interactive NYC map interface allows users to select pickup and dropoff locations and instantly receive fare and distance predictions from the model. Planned additions include spatial heatmaps, prediction error visualization, and temporal filtering.

---

## Data Sources
NYC TLC trip record downloads: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page <br>

After downloading and preprocessing the data, the local project keeps it under `data/` in this layout:

```text
data/
├── concate_data/
│   ├── green_taxi_2025/
│   └── yellow_taxi_2025/
├── taxi_zones/
│   ├── taxi_zones.shp
│   ├── taxi_zones.shx
│   ├── taxi_zones.dbf
│   ├── taxi_zones.cpg
│   └── taxi_zones.prj
└── taxi_zone_centroids.csv
```

The app reads the centroid file at `data/taxi_zone_centroids.csv`, and the notebook pipeline uses the raw taxi-zone shapefile data in `data/taxi_zones/` plus the trip files in `data/concate_data/`.

If you are starting from a fresh download, place the raw TLC parquet files into the matching year folders under `data/concate_data/` and keep the taxi zone shapefiles in `data/taxi_zones/` so the notebook and local API can resolve pickup/dropoff locations correctly.<br>

## AutoGluon Installation: https://auto.gluon.ai/stable/install.html <br>
conda create -n autogluon-win python=3.10 -y<br>
conda activate autogluon-win<br>

pip install -U pip<br>
pip install -U setuptools wheel<br>
pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu<br>


# NYC Taxi Local App

FastAPI backend plus Flask frontend for the AutoGluon taxi prediction models trained in the notebooks.

## Setup

```bash
# 1. Create and activate your environment
conda create -n autogluon-win python=3.10 -y
conda activate autogluon-win

# 2. Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install flask requests

# 3. Start the API server
python -m uvicorn local_api:app --reload --port 8000

# 4. In a second terminal, start the UI server
python web_server.py
```

The backend runs at http://127.0.0.1:8000 and the UI runs at http://127.0.0.1:5001.

---

## Project Structure

Run the servers from the project root so the model folders and data paths resolve correctly:

```text
project-root/
├── local_api.py
├── web_server.py
├── templates/
│   └── index.html
├── AutogluonModels/
│   ├── taxi_duration_quick/
│   └── taxi_fare_quick/
└── data/
    └── taxi_zone_centroids.csv
```

## Verify It Is Running

Open these URLs after starting the API:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Expected health response:

```json
{ "status": "online", "models_ready": ["duration", "fare"] }
```

Then open the UI at http://127.0.0.1:5001/

---

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check whether the API is online |
| GET | `/zones/nearby` | Resolve the nearest taxi zone for a latitude/longitude pair |
| POST | `/predict` | Predict trip duration and fare for one or more trips |

## Example API Call

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trips": [{
      "trip_distance": 2.5,
      "pu_location_id": 161,
      "do_location_id": 236
    }]
  }'
```

Expected response shape:

```json
{
  "results": [
    {
      "predicted_duration_minutes": 0.0,
      "predicted_fare_amount": 0.0,
      "pickup_zone": "...",
      "dropoff_zone": "..."
    }
  ],
  "count": 1
}
```

## Notes

- The AutoGluon models are already trained in `AutoML.ipynb`; there is no separate training step in the local startup flow.
- The backend loads the duration and fare predictors lazily on first request.
- CORS is open (`*`) for local development.