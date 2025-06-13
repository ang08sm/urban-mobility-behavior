import pandas as pd
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"


def build_od_flows():
    """
    From the merged CDR data (user_id, tower_id, timestamp),
    compute origin→destination(od) flows per hour.

    - Loads data/processed/cdr_merged.parquet
    - Sorts by user_id + timestamp
    - For each user, identifies consecutive tower hops (prev_tower -> current tower)
    - Aggregates counts of (hour, origin, destination) into od_flows.parquet
    """
    cdr_path = PROCESSED_DIR / "cdr_merged.parquet"
    if not cdr_path.exists():
        print(f"File not found: {cdr_path}")
        return

    print("Loading merged CDR data...")
    df = pd.read_parquet(cdr_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")

    # Sort by user_id & timestamp
    df.sort_values(["user_id", "timestamp"], inplace=True)

    # Determine origin/destination tower for each event
    df["origin_tower"] = df.groupby("user_id")["tower_id"].shift()
    df["dest_tower"] = df["tower_id"]

    print("Total records before dropna:", df.shape[0])
    print("Unique users:", df['user_id'].nunique())
    print("CDRs per user (sample):")
    print(df.groupby("user_id").size().describe())

    df = df.dropna(subset=["origin_tower"])

    # Floor timestamp to the hour
    df["hour"] = df["timestamp"].dt.floor("h")

    # Count flows per (hour, origin, dest)
    od = (
        df.groupby(["hour", "origin_tower", "dest_tower"])
            .size()
            .reset_index(name="count")
    )

    out_path = PROCESSED_DIR / "od_flows.parquet"
    od.to_parquet(out_path, index=False)
    print(f"OD flows saved to {out_path.name} (rows: {od.shape[0]})")


def build_gps_speed_features():
    """
    From merged GPS data (device_id, latitude, longitude, timestamp, speed_kmph),
    aggregate average speed per hour + coarse grid cell:
    - Loads data/processed/gps_merged.parquet
    - Floors timestamp to hour
    - Rounds lat/lon to 2 decimals to create grid cell
    - Aggregates mean speed per (hour, grid_id)
    """
    gps_path = PROCESSED_DIR / "gps_merged.parquet"
    if not gps_path.exists():
        print(f"File not found: {gps_path}")
        return

    print("Loading merged GPS data...")
    df = pd.read_parquet(gps_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")

    # Floor timestamp to hour
    df["hour"] = df["timestamp"].dt.floor("h")

    # Create coarse grid cell by rounding lat/lon
    df["grid_lat"] = df["latitude"].round(2)
    df["grid_lon"] = df["longitude"].round(2)
    df["grid_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    # Aggregate average speed_kmph per (hour, grid_id)
    gps_speed = (
        df.groupby(["hour", "grid_id"])["speed_kmph"]
            .mean()
            .reset_index()
            .rename(columns={"speed_kmph": "avg_speed_kmph"})
    )

    out_path = PROCESSED_DIR / "gps_speed_features.parquet"
    gps_speed.to_parquet(out_path, index=False)
    print(f"GPS speed features saved to {out_path.name} (rows: {gps_speed.shape[0]})")


if __name__ == "__main__":
    build_od_flows()
    build_gps_speed_features()