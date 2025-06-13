import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "cdr").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "gps").mkdir(parents=True, exist_ok=True)


def simulate_cdr_data(num_users=10000, num_records=3000000):
    print(f"Simulating {num_records:,} CDRs for {num_users:,} users...")

    user_ids = np.random.choice(range(1, num_users + 1), size=num_records, replace=True)

    base_time = datetime(2024, 5, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=random.randint(0, 7 * 24 * 3600)) for _ in range(num_records)]

    tower_ids = [f"T{random.randint(1, 50)}" for _ in range(num_records)]

    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": pd.to_datetime(timestamps),
        "tower_id": tower_ids
    })

    raw_path = RAW_DIR / f"cdr_lucknow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw CDR saved to {raw_path.name} (shape={df.shape})")

    # Chunk & save as parquet
    chunk_size = 200000
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start + chunk_size]
        out_path = PROCESSED_DIR / "cdr" / f"cdr_chunk_{i:03}.parquet"
        chunk.to_parquet(out_path, index=False)
    print(f"{i+1} CDR chunks written to data/processed/cdr/")


def simulate_gps_data(num_devices=3000, num_records=500000):
    print(f"Simulating {num_records:,} GPS records for {num_devices:,} devices...")

    device_ids = np.random.choice(range(1, num_devices + 1), size=num_records, replace=True)

    base_time = datetime(2024, 5, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=random.randint(0, 7 * 24 * 3600)) for _ in range(num_records)]

    latitudes = np.random.uniform(26.80, 26.92, num_records)
    longitudes = np.random.uniform(80.90, 81.02, num_records)
    speeds = np.random.normal(loc=30, scale=10, size=num_records).clip(0, 100)  # kmph

    df = pd.DataFrame({
        "device_id": device_ids,
        "timestamp": pd.to_datetime(timestamps),
        "latitude": latitudes,
        "longitude": longitudes,
        "speed_kmph": speeds
    })

    raw_path = RAW_DIR / f"gps_lucknow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw GPS saved to {raw_path.name} (shape={df.shape})")

    # Chunk & save as parquet
    chunk_size = 100000
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start + chunk_size]
        out_path = PROCESSED_DIR / "gps" / f"gps_chunk_{i:03}.parquet"
        chunk.to_parquet(out_path, index=False)
    print(f"✓ {i+1} GPS chunks written to data/processed/gps/")


if __name__ == "__main__":
    simulate_cdr_data()
    simulate_gps_data()