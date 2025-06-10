import pandas as pd
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

def merge_cdr_chunks():
    """
    Reads all Parquet files under data/processed/cdr_cleaned/,
    concatenates them into one DataFrame, and writes to data/processed/cdr_merged.parquet
    """
    cdr_folder = PROCESSED_DIR / "cdr_cleaned"
    if not cdr_folder.exists():
        print(f"Directory not found: {cdr_folder}")
        return

    parquet_files = sorted(cdr_folder.glob("*.parquet"))
    if not parquet_files:
        print(f"No CDR chunks found under: {cdr_folder}")
        return

    print(f"Merging {len(parquet_files)} CDR chunk(s)...")
    df_list = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        df_list.append(df)
    full_cdr = pd.concat(df_list, ignore_index=True)
    out_path = PROCESSED_DIR / "cdr_merged.parquet"
    full_cdr.to_parquet(out_path, index=False)
    print(f"Written merged CDR to {out_path.name} (shape {full_cdr.shape})")


def merge_gps_chunks():
    """
    Reads all Parquet files under data/processed/gps_cleaned/,
    concatenates them, and writes to data/processed/gps_merged.parquet
    """
    gps_folder = PROCESSED_DIR / "gps_cleaned"
    if not gps_folder.exists():
        print(f"Directory not found: {gps_folder}")
        return

    parquet_files = sorted(gps_folder.glob("*.parquet"))
    if not parquet_files:
        print(f"No GPS chunks found under: {gps_folder}")
        return

    print(f"Merging {len(parquet_files)} GPS chunk(s)...")
    df_list = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        df_list.append(df)
    full_gps = pd.concat(df_list, ignore_index=True)
    out_path = PROCESSED_DIR / "gps_merged.parquet"
    full_gps.to_parquet(out_path, index=False)
    print(f"Written merged GPS to {out_path.name} (shape {full_gps.shape})")


if __name__ == "__main__":
    merge_cdr_chunks()
    merge_gps_chunks()