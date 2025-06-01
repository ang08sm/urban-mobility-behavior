import os
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

# Chuck size
CHUNK_SIZE = 200_000

def ensure_processed_folders_exist():
    """Make sure data/processed/cdr/ and data/processed/gps/ exist."""
    (PROCESSED_DIR / "cdr").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "gps").mkdir(parents=True, exist_ok=True)

def stream_csv_to_parquet(raw_file: Path, kind: str):
    """
    Read raw_file (CSV) in CHUNK_SIZE pieces, and write each chunk as
    a Parquet file under data/processed/{kind}/ or data/streamed/{kind}/.

    kind must be either "cdr" or "gps".
    """
    assert kind in ("cdr", "gps"), "kind must be 'cdr' or 'gps'"

    print(f"\n▶ Streaming {kind.upper()} file: {raw_file.name}")

    # Cleaned Parquet directly into data/processed/{type}/
    dest_folder = PROCESSED_DIR / kind

    for idx, chunk in enumerate(pd.read_csv(raw_file, chunksize=CHUNK_SIZE, iterator=True)):
        # Optionally: if there is transform logic that expects a Parquet path,
        # it could first write this chunk to STREAMED_DIR, then call transform_chunk(...)
        # For simplicity,it'll be directly clean & write to data/processed/

        # Example “light cleaning” in‐line:
        if kind == "cdr":
            # Parse timestamp (string → datetime)
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True)
        else:  # kind == "gps"
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True)
            # Optionally clamp lat/lon to Lucknow bounds:
            chunk = chunk[
                chunk["latitude"].between(26.80, 26.92) &
                chunk["longitude"].between(80.90, 81.02)
            ]

        # Write each chunk as Parquet
        chunk_name = raw_file.stem 
        out_path = dest_folder / f"{chunk_name}_chunk_{idx:03d}.parquet"
        chunk.to_parquet(out_path, index=False)
        print(f"  • Wrote chunk #{idx:03d} → {out_path.name}")

    print(f"Done streaming {raw_file.name} → {len(list(dest_folder.glob(f'{chunk_name}_chunk_*.parquet')))} files")


if __name__ == "__main__":
    """
    Usage:
    python src/data_pipeline/streamer.py

    This script will:
    1) Look in data/raw/ for any CSVs named cdr_*.csv or gps_*.csv
    2) For each, read in CHUNK_SIZE rows at a time and write cleaned Parquet
    into data/processed/cdr/ or data/processed/gps/
    """
    ensure_processed_folders_exist()

    # 1) Find all "cdr_*.csv"
    cdr_files = sorted(RAW_DIR.glob("cdr_lucknow_*.csv"))
    # 2) Find all "gps_*.csv"
    gps_files = sorted(RAW_DIR.glob("gps_lucknow_*.csv"))

    if not cdr_files and not gps_files:
        print("No raw CDR/GPS CSVs found under data/raw/. Run ingest.py first.")
        exit(0)

    # Stream each CDR CSV
    for csv_path in cdr_files:
        stream_csv_to_parquet(csv_path, kind="cdr")

    # Stream each GPS CSV
    for csv_path in gps_files:
        stream_csv_to_parquet(csv_path, kind="gps")

    print("\n Streaming & basic-cleaning complete. Check data/processed/ for Parquet files.")