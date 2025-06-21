import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def create_time_features(df, time_col='hour'):
    """
    Extracts cyclical and basic time-based features from a datetime column.
    These features help models understand time patterns (e.g., hourly, daily, weekly cycles).
    """
    df[time_col] = pd.to_datetime(df[time_col]) 
    df['hour_of_day'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek 
    df['day_of_year'] = df[time_col].dt.dayofyear
    df['month'] = df[time_col].dt.month
    df['week_of_year'] = df[time_col].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) 

    # Cyclical features for hour of day and day of week using sine/cosine transformations
    # This helps models capture the cyclical nature without implying a linear relationship
    # between, for example, hour 23 and hour 0.
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df



def create_lag_and_rolling_features(df, group_cols, time_col, target_col, lags=[1, 24, 48, 168], window_sizes=[3, 6, 24]):
    """
    Creates lag and rolling mean/std features for a target column.
    Features are generated independently for each group (e.g., each grid_id or OD pair).

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_cols (list): Columns to group by before creating features (e.g., ['grid_id'] or ['origin_tower', 'dest_tower']).
        time_col (str): Name of the datetime column.
        target_col (str): Name of the column for which to create features (e.g., 'avg_speed_kmph' or 'count').
        lags (list): List of integers representing time lags (in hours/time steps) for lag features.
        window_sizes (list): List of integers representing window sizes for rolling mean/std features.
    """
    # Sort data by group and time to ensure correct lag/rolling calculations
    df = df.sort_values(by=group_cols + [time_col])

    # Lag features: Value of the target at previous time steps
    print(f"  - Creating lag features for '{target_col}'...")
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)

    # Rolling window features: Statistics over a preceding window of data
    print(f"  - Creating rolling features for '{target_col}'...")
    for window in window_sizes:
        # min_periods=1 allows calculation even with fewer data points than window size at the start of a series
        df[f'{target_col}_rolling_mean_{window}h'] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'{target_col}_rolling_std_{window}h'] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).std())

    # Fill NaNs created by lagging/rolling.
    # Forward fill (ffill) propagates last valid observation forward.
    # Backward fill (bfill) propagates next valid observation backward (for initial NaNs).
    # This is a common strategy, but you might explore more sophisticated imputation
    # (e.g., mean of the series, or using a predictive model for imputation) if needed.
    print("  - Filling NaNs introduced by lag/rolling features...")
    for col in df.columns:
        if col.startswith(f'{target_col}_lag_') or col.startswith(f'{target_col}_rolling_'):
            df[col] = df.groupby(group_cols)[col].ffill().bfill()
            # After ffill/bfill within groups, if any NaNs remain (e.g., an entire group is NaN for that feature),
            # fill with a sensible default like 0 or the overall mean/median.
            df[col] = df[col].fillna(0) # Using 0 as a placeholder for initial missing lags/rolling values

    return df

def create_od_prediction_features():
    """
    Loads 'od_flows.parquet', adds advanced time-series features (time-based, lags, rolling),
    and saves the enriched dataset as 'od_flows_engineered.parquet'.
    """
    od_path = PROCESSED_DIR / "od_flows.parquet"
    if not od_path.exists():
        print(f"File not found: {od_path}. Please run feature_engineering.py first.")
        return

    print("Creating features for OD flow prediction...")
    od_df = pd.read_parquet(od_path)

    # 1. Create general time-based features
    od_df = create_time_features(od_df, time_col='hour')

    # 2. Create lag and rolling features for 'count'
    # Grouping by both origin_tower and dest_tower ensures features are specific to each unique flow path.
    group_cols_od = ['origin_tower', 'dest_tower']
    target_col_od = 'count'
    od_df = create_lag_and_rolling_features(od_df, group_cols_od, 'hour', target_col_od)

    # Save the enriched dataset
    out_path_od = PROCESSED_DIR / "od_flows_engineered.parquet"
    od_df.to_parquet(out_path_od, index=False)
    print(f"Engineered OD flow features saved to {out_path_od.name} (shape={od_df.shape})")
    print("\nFirst 5 rows of engineered OD flows:")
    print(od_df.head())

def create_gps_speed_prediction_features():
    """
    Loads 'gps_speed_features.parquet', adds advanced time-series features (time-based, lags, rolling),
    and saves the enriched dataset as 'gps_speed_features_engineered.parquet'.
    """
    gps_path = PROCESSED_DIR / "gps_speed_features.parquet"
    if not gps_path.exists():
        print(f"File not found: {gps_path}. Please run feature_engineering.py first.")
        return

    print("Creating features for GPS speed prediction...")
    gps_df = pd.read_parquet(gps_path)

    # 1. Create general time-based features
    gps_df = create_time_features(gps_df, time_col='hour')

    # 2. Create lag and rolling features for 'avg_speed_kmph'
    # Grouping by 'grid_id' ensures features are specific to each geographical grid.
    group_cols_gps = ['grid_id']
    target_col_gps = 'avg_speed_kmph'
    gps_df = create_lag_and_rolling_features(gps_df, group_cols_gps, 'hour', target_col_gps)

    # Save the enriched dataset
    out_path_gps = PROCESSED_DIR / "gps_speed_features_engineered.parquet"
    gps_df.to_parquet(out_path_gps, index=False)
    print(f"Engineered GPS speed features saved to {out_path_gps.name} (shape={gps_df.shape})")
    print("\nFirst 5 rows of engineered GPS speed features:")
    print(gps_df.head())


if __name__ == "__main__":
    # Ensure PROCESSED_DIR exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    create_od_prediction_features()
    print("\n" + "="*50 + "\n") # Separator
    create_gps_speed_prediction_features()