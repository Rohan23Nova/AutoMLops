from preprocess import load_and_save_data
from train import train_models
from model_selection import select_and_save_best_model
from drift_detection import detect_drift
import pandas as pd
from monitoring import log_event


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATASET = "data/processed/iris.csv"
DEFAULT_TARGET = "target"


# ============================================================
# RETRAINING / DRIFT DETECTION
# ============================================================
def check_and_retrain(
    reference_path="data/processed/reference.csv",
    current_path="data/processed/current.csv",
    target_column="target",
    auto_deploy=True
):
    try:
        reference = pd.read_csv(reference_path)
        current = pd.read_csv(current_path)

        report = detect_drift(reference, current)

        drift_found = any(
            result["drift_detected"]
            for result in report.values()
        )

        log_event(
            "drift_check",
            {
                "drift_detected": drift_found,
                "report": report
            }
        )

        if not drift_found:
            print("No drift detected.")

            log_event(
                "retraining_skipped",
                {
                    "reason": "No data drift detected"
                }
            )

            return {
                "status": "no_drift",
                "drift_detected": False,
                "retrained": False
            }

        print("Drift detected. Retraining model...")

        log_event(
            "retraining_started",
            {
                "dataset": current_path,
                "target_column": target_column
            }
        )

        result = run_pipeline(
            dataset_path=current_path,
            target_column=target_column,
            auto_deploy=auto_deploy
        )

        log_event(
            "retraining_completed",
            {
                "dataset": current_path,
                "target_column": target_column,
                "pipeline_result": result
            }
        )

        return {
            "status": "retrained",
            "drift_detected": True,
            "retrained": True,
            "pipeline_result": result
        }

    except Exception as e:

        log_event(
            "retraining_failed",
            {
                "error": str(e)
            },
            status="failed"
        )

        print(f"Retraining failed: {e}")

        return {
            "status": "failed",
            "error": str(e)
        }
# ============================================================
# MAIN AUTOMATED ML PIPELINE
# ============================================================

def run_pipeline(
    dataset_path=None,
    target_column=None,
    auto_deploy=False
):
    if dataset_path is None:
        load_and_save_data()
        dataset_path = DEFAULT_DATASET

    if target_column is None:
        target_column = DEFAULT_TARGET

    results = train_models(
        data_path=dataset_path,
        target_column=target_column
    )

    best_model = select_and_save_best_model(
        results,
        target_column=target_column,
        dataset_path=dataset_path
    )

    deployment = None

    if auto_deploy:
        from deploy_model import deploy_model

        deployment = deploy_model()

        print("\nModel deployed successfully:")
        print(f"Model: {deployment['model_name']}")
        print(f"Version: {deployment['model_version']}")

    print("\nPipeline execution completed.")

    return {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "deployment": deployment
    }