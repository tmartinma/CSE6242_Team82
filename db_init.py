import gc
import os
import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

# --- CONFIGURATION ---
# NOTE: HAVE LOCAL MEMORY ISSUES WITH YELLOW TAXI DATA. USE GREEN IN MVP FOR NOW.
folders = {"yellow": "./data/yellow_taxi_2025"}
folders = {"green": "./data/green_taxi_2025"}
shapefile_path = "./data/taxi_zones/taxi_zones.shp"
db_name = "local_db.db"


def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    # These settings prevent the "Freeze" by moving processing to RAM and skipping undo-logs
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -2000000;")  # 2GB Cache
    return conn


def get_zones_lookup(shp_path):
    print("Loading Taxi Zones...")
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf["longitude"] = gdf.geometry.centroid.x
    gdf["latitude"] = gdf.geometry.centroid.y
    gdf["geometry"] = gdf["geometry"].astype(str)
    return pd.DataFrame(gdf)


def get_master_columns(folders_dict):
    """Scan files to build a universal column list."""
    all_cols = set()
    for folder in folders_dict.values():
        files = list(Path(folder).glob("*.parquet"))
        if files:
            schema = pq.ParquetFile(files[0]).schema.names
            all_cols.update(schema)
    all_cols.update(["pickup_datetime", "dropoff_datetime"])
    return sorted(list(all_cols))


def process_single_file(file_path, conn, zones_df, master_cols):
    print(f"\nProcessing: {file_path.name}")
    p_file = pq.ParquetFile(file_path)
    schema = p_file.schema.names

    p_src = [c for c in schema if "pickup_datetime" in c][0]
    d_src = [c for c in schema if "dropoff_datetime" in c][0]

    # Smaller batches (15k) are better for Yellow taxi data
    for batch in p_file.iter_batches(batch_size=15000):
        df = batch.to_pandas()

        # 1. Standardize
        df["pickup_datetime"] = df[p_src]
        df["dropoff_datetime"] = df[d_src]

        # 2. Join
        joined = df.merge(
            zones_df, left_on="PULocationID", right_on="LocationID", how="left"
        )

        # 3. Align to Master Schema
        # This adds missing columns as NULLs so the DB doesn't crash
        final_cols = master_cols + [c for c in zones_df.columns if c not in master_cols]
        joined = joined.reindex(columns=final_cols)

        # 4. FAST INSERT: Using native SQL instead of SQLAlchemy/Pandas overhead
        # We wrap this in a manual transaction
        joined.to_sql("final_combined_data", conn, if_exists="append", index=False)

        del df, joined
        gc.collect()
        print(".", end="", flush=True)


if __name__ == "__main__":
    if os.path.exists(db_name):
        os.remove(db_name)

    db_conn = setup_db(db_name)
    zones = get_zones_lookup(shapefile_path)
    master_columns = get_master_columns(folders)

    for taxi_type, folder_path in folders.items():
        path = Path(folder_path)
        if not path.exists():
            continue

        for f in path.glob("*.parquet"):
            # Process one file at a time
            process_single_file(f, db_conn, zones, master_columns)
            # Commit after EVERY file to clear the memory buffer
            db_conn.commit()
            print(f"\nSaved {f.name} to disk.")

    print("\nCreating Final Indexes...")
    db_conn.execute("CREATE INDEX idx_pdt ON final_combined_data (pickup_datetime);")
    db_conn.close()
    print("Success! No more freezing.")
