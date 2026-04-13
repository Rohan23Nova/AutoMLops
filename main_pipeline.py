from preprocess import load_and_save_data
from train import train_models
from model_selection import select_and_save_best_model
from drift_detection import detect_drift
import pandas as pd

def check_and_retrain():
    reference = pd.read_csv("data/processed/reference.csv")
    current = pd.read_csv("data/processed/current.csv")

    report = detect_drift(reference, current)

    drift_found = any(col["drift_detected"] for col in report.values())

    if drift_found:
        print("⚠️ Drift detected. Retraining model...")
        run_pipeline()
    else:
        print("✅ No drift detected.")
def run_pipeline():
    load_and_save_data()
    results = train_models()
    select_and_save_best_model(results)
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()