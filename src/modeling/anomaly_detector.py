import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

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
    print(f"✓ Hourly anomalies saved to {out_csv.name}")

    try:
        import joblib
        model_path = PROCESSED_DIR / "isolation_forest_model.pkl"
        joblib.dump(iso, model_path)
        print(f"IsolationForest model saved to {model_path.name}")
    except ImportError:
        print("⚠️  joblib not installed; skipping model save.")

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


if __name__ == "__main__":
    detect_hourly_demand_anomalies()