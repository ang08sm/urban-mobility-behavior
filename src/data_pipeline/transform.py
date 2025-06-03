import pandas as pd
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"

def transform_cdr_chunk(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df[df["tower_id"].notnull()]
    cleaned_folder = PROCESSED_DIR / "cdr_cleaned"
    cleaned_folder.mkdir(parents=True, exist_ok=True)
    out_path = cleaned_folder / f"{parquet_path.stem}_cleaned.parquet"
    df.to_parquet(out_path, index=False)
    print(f"CDR chunk cleaned → {out_path.name}")

def transform_gps_chunk(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    df = df[df["latitude"].between(26.80, 26.92) & df["longitude"].between(80.90, 81.02)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    cleaned_folder = PROCESSED_DIR / "gps_cleaned"
    cleaned_folder.mkdir(parents=True, exist_ok=True)
    out_path = cleaned_folder / f"{parquet_path.stem}_cleaned.parquet"
    df.to_parquet(out_path, index=False)
    print(f"GPS chunk cleaned → {out_path.name}")


if __name__ == "__main__":
    cdr_chunks = list((PROCESSED_DIR / "cdr").glob("*.parquet"))
    gps_chunks = list((PROCESSED_DIR / "gps").glob("*.parquet"))

    if not cdr_chunks and not gps_chunks:
        print("No CDR or GPS chunks found in processed folders.")
    else:
        for path in cdr_chunks:
            transform_cdr_chunk(path)
        for path in gps_chunks:
            transform_gps_chunk(path)
