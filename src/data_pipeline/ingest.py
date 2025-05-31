import os
import uuid
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from shutil import copyfile
from pathlib import Path

N_CDR_RECORDS = 3_000_000    
N_GPS_RECORDS = 5_000_000

# Destination folder for “raw” data
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR   = PROJECT_ROOT / "data" / "raw"
TIMEZONE       = pytz.timezone("Asia/Kolkata")


def ensure_raw_folder_exists():
    """Make sure data/raw/ exists."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _lucknow_tower_locations():
    """
    Return a list of (tower_id, lat, lon) for synthetic Lucknow towers.
    In a real‐world project, you could pull these from an Open Cell ID database or an API.
    """
    towers = [
        ("TWR_LUCKNOW_CENTER",   26.846733, 80.946211),
        ("TWR_INDIRA_TOWERS",    19.970052, 73.780937),
        ("TWR_SRIRAM_TOWERS",    26.850452, 80.948221),
        ("TWR_AVADH_TOWER",      26.849160, 80.944260),
        ("TWR_GARIB_CORP",       26.871114, 80.972644),
    ]
    return towers


def generate_synthetic_cdr(output_path: Path, n_records: int):
    """
    Generates a CSV of CDR records. Columns:
    - user_id, tower_id, timestamp, event_type
    - user_id: random UUID4
    - tower_id: one of 20 synthetic Lucknow towers
    - timestamp: a random datetime in the last 7 days
    - event_type: one of ["CALL_START","CALL_END","SMS_SEND","SMS_RECV","DATA"]
    """
    towers = _lucknow_tower_locations()
    tower_ids = [t[0] for t in towers]
    event_types = ["CALL_START", "CALL_END", "SMS_SEND", "SMS_RECV", "DATA"]

    chunk_size = 200_000
    rows_written = 0
    now = datetime.now(TIMEZONE)
    seven_days_ago = now - timedelta(days=7)

    # Header & data in incremental chunks to avoid huge memory usage
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("user_id,tower_id,timestamp,event_type\n")
        while rows_written < n_records:
            batch_size = min(chunk_size, n_records - rows_written)

            # Generates fields in arrays/lists
            user_ids   = [str(uuid.uuid4()) for _ in range(batch_size)]
            tower_picks = np.random.choice(tower_ids, size=batch_size)
            timestamps = [
                (seven_days_ago + timedelta(seconds=random.randint(0, 7 * 24 * 3600)))
                .strftime("%Y-%m-%d %H:%M:%S")
                for _ in range(batch_size)
            ]
            events = np.random.choice(event_types, size=batch_size)

            # Each record
            for uid, twr, ts, ev in zip(user_ids, tower_picks, timestamps, events):
                f.write(f"{uid},{twr},{ts},{ev}\n")

            rows_written += batch_size
            print(f"  > CDR rows generated: {rows_written}/{n_records}", end="\r")

    print(f"\n CDR CSV written: {output_path}")


def generate_synthetic_gps(output_path: Path, n_records: int):
    """
    Generates a CSV of synthetic GPS traces. Columns:
    - device_id, latitude, longitude, timestamp, speed_kmph
    - device_id: “DEV_xxxxx” (simulate ~5000 devices)
    - latitude/longitude: random point inside bounding box for Lucknow
    - timestamp: a random datetime in the last 7 days
    - speed_kmph: random float 0–80
    """
    LAT_MIN, LAT_MAX = 26.80, 26.92
    LON_MIN, LON_MAX = 80.90, 81.02
    chunk_size = 200_000
    rows_written = 0
    now = datetime.now(TIMEZONE)
    seven_days_ago = now - timedelta(days=7)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("device_id,latitude,longitude,timestamp,speed_kmph\n")
        while rows_written < n_records:
            batch_size = min(chunk_size, n_records - rows_written)
            device_ids = [f"DEV_{random.randint(1,5000):05d}" for _ in range(batch_size)]
            lats = np.random.uniform(LAT_MIN, LAT_MAX, size=batch_size)
            lons = np.random.uniform(LON_MIN, LON_MAX, size=batch_size)
            timestamps = [
                (seven_days_ago + timedelta(seconds=random.randint(0, 7 * 24 * 3600)))
                .strftime("%Y-%m-%d %H:%M:%S")
                for _ in range(batch_size)
            ]
            speeds = np.random.uniform(0, 80, size=batch_size).round(2)

            for did, la, lo, ts, sp in zip(device_ids, lats, lons, timestamps, speeds):
                f.write(f"{did},{la:.6f},{lo:.6f},{ts},{sp}\n")

            rows_written += batch_size
            print(f"  > GPS rows generated: {rows_written}/{n_records}", end="\r")

    print(f"\n✓ Synthetic GPS CSV written: {output_path}")


def ingest_real_csv(source_path: Path):
    """
    If you already have a “real” CDR/GPS CSV lying around, call this function to
    copy it into data/raw/. Example usage:
        ingest_real_csv(Path("/home/user/downloads/my_cdr.csv"))
    """
    dest = RAW_DATA_DIR / source_path.name
    copyfile(source_path, dest)
    print(f"✓ Copied real CSV → {dest}")


if __name__ == "__main__":
    """
    Usage:
        (1) To generate synthetic data, just run this script without args:
            python src/data_pipeline/ingest.py
        It will produce:
            data/raw/cdr_lucknow_<timestamp>.csv
            data/raw/gps_lucknow_<timestamp>.csv

        (2) To ingest a pre‐downloaded real CSV, edit the ingest_real_csv(...) call below
            or pass a path via an environment variable / command‐line argument.
    """
    ensure_raw_folder_exists()
    timestamp_str = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")

    # 1) Synthetic CDR
    cdr_filename = f"cdr_lucknow_{timestamp_str}.csv"
    cdr_path = RAW_DATA_DIR / cdr_filename
    print("Generating synthetic CDR data ...")
    generate_synthetic_cdr(cdr_path, N_CDR_RECORDS)

    # 2) Synthetic GPS
    gps_filename = f"gps_lucknow_{timestamp_str}.csv"
    gps_path = RAW_DATA_DIR / gps_filename
    print("\nGenerating synthetic GPS data ...")
    generate_synthetic_gps(gps_path, N_GPS_RECORDS)

    # 3) (Optional) If you have a real CSV, call ingest_real_csv(...) here, e.g.:
    # real_cdr = Path("/home/you/downloads/cdr_real.csv")
    # ingest_real_csv(real_cdr)

    print("\nIngestion complete. Check data/raw/ for new CSV files.")