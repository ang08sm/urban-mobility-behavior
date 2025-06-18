import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

def detect_hourly_demand_anomalies():
    """
    Loads od_flows.parquet, aggregates to total hourly demand,
    then fits an IsolationForest to flag anomalous hours.
    Saves:
        - hourly_anomalies.csv
        - isolation_forest_model.pkl (if joblib available)
        - hourly_anomalies_plot.png
    """
    od_path = PROCESSED_DIR / "od_flows.parquet"
    if not od_path.exists():
        print(f" File not found: {od_path}")
        return

    print(" Loading OD flows...")
    od = pd.read_parquet(od_path)
    od["hour"] = pd.to_datetime(od["hour"])

    # Aggregate
    hourly = od.groupby("hour")["count"].sum().reset_index(name="total_count")
    hourly.sort_values("hour", inplace=True)
    hourly.reset_index(drop=True, inplace=True)

    # Fit IsolationForest
    X = hourly[["total_count"]]
    print("Fitting IsolationForest for anomaly detection...")
    iso = IsolationForest(contamination=0.05, random_state=42)
    hourly["anomaly_score"] = iso.fit_predict(X)  # +1 normal, -1 anomaly
    hourly["is_anomaly"] = hourly["anomaly_score"] == -1

    # Save CSV
    out_csv = PROCESSED_DIR / "hourly_anomalies.csv"
    hourly.to_csv(out_csv, index=False)
    print(f"Hourly anomalies saved to {out_csv.name}")

    try:
        import joblib
        model_path = PROCESSED_DIR / "isolation_forest_model.pkl"
        joblib.dump(iso, model_path)
        print(f"IsolationForest model saved to {model_path.name}")
    except ImportError:
        print("joblib not installed; skipping model save.")

    # Plot
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(hourly["hour"], hourly["total_count"], label="Total Demand")
        anomalies = hourly[hourly["is_anomaly"]]
        plt.scatter(anomalies["hour"], anomalies["total_count"], color="red", label="Anomaly")
        plt.xlabel("Hour")
        plt.ylabel("Total OD Count")
        plt.title("Hourly Demand with Anomalies Flagged")
        plt.legend()
        plt.tight_layout()
        plot_path = PROCESSED_DIR / "hourly_anomalies_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"✓ Saved anomaly plot to {plot_path.name}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

def detect_gps_speed_anomalies():
    """
    Loads gps_speed_features.parquet, extracts time-based features,
    then fits an IsolationForest to flag anomalous average speeds.
    Saves:
        - gps_speed_anomalies.csv
        - gps_isolation_forest_model.pkl
        - gps_speed_anomalies_plot.png
    """
    gps_path = PROCESSED_DIR / "gps_speed_features.parquet"
    if not gps_path.exists():
        print(f"File not found: {gps_path}")
        return

    print("Loading GPS speed features...")
    gps_speeds = pd.read_parquet(gps_path)
    gps_speeds["hour"] = pd.to_datetime(gps_speeds["hour"])

    gps_speeds['hour_of_day'] = gps_speeds['hour'].dt.hour
    gps_speeds['day_of_week'] = gps_speeds['hour'].dt.dayofweek
    gps_speeds['is_weekend'] = gps_speeds['hour'].dt.dayofweek.isin([5, 6]).astype(int)

    # Features for IsolationForest. Including time-based features
    # helps detect anomalies specific to time context (e.g., unusually low speed
    # during peak hours, or unusually high speed during off-peak hours).
    features_for_anomaly = ['avg_speed_kmph', 'hour_of_day', 'day_of_week', 'is_weekend']
    X_gps = gps_speeds[features_for_anomaly]

    print("Fitting IsolationForest for anomaly detection on GPS speeds...")
    iso_gps = IsolationForest(contamination=0.01, random_state=42) 
    gps_speeds["anomaly_score"] = iso_gps.fit_predict(X_gps)  
    gps_speeds["is_anomaly"] = gps_speeds["anomaly_score"] == -1

    # Save CSV
    out_csv_gps = PROCESSED_DIR / "gps_speed_anomalies.csv"
    gps_speeds.to_csv(out_csv_gps, index=False)
    print(f"GPS speed anomalies saved to {out_csv_gps.name}")

    # Save model
    model_path_gps = PROCESSED_DIR / "gps_isolation_forest_model.pkl"
    joblib.dump(iso_gps, model_path_gps)
    print(f"GPS IsolationForest model saved to {model_path_gps.name}")

    try:
        plt.figure(figsize=(12, 6))
        
        plt.scatter(gps_speeds["hour"], gps_speeds["avg_speed_kmph"],
                    c=gps_speeds["is_anomaly"].map({True: 'red', False: 'blue'}),
                    label="Normal" if not gps_speeds["is_anomaly"].any() else None,
                    alpha=0.6, s=10)
        
        if gps_speeds["is_anomaly"].any():
            anomalies_gps = gps_speeds[gps_speeds["is_anomaly"]]
            plt.scatter(anomalies_gps["hour"], anomalies_gps["avg_speed_kmph"],
                        color="red", label="Anomaly", alpha=0.8, s=20, marker='x')

        plt.xlabel("Hour")
        plt.ylabel("Average Speed (km/h)")
        plt.title("GPS Average Speed with Anomalies Flagged")
        plt.legend()
        plt.tight_layout()
        plot_path_gps = PROCESSED_DIR / "gps_speed_anomalies_plot.png"
        plt.savefig(plot_path_gps)
        plt.close()
        print(f"Saved GPS speed anomaly plot to {plot_path_gps.name}")
    except Exception as e:
        print(f"Could not generate GPS plot: {e}")

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    detect_hourly_demand_anomalies()
    print("\n" + "="*50 + "\n")
    detect_gps_speed_anomalies()