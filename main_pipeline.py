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
def check_and_retrain():

    reference = pd.read_csv(
        "data/processed/reference.csv"
    )

    current = pd.read_csv(
        "data/processed/current.csv"
    )

    report = detect_drift(
        reference,
        current
    )

    drift_found = any(
        col["drift_detected"]
        for col in report.values()
    )

    log_event(
        "drift_check",
        report
    )

    # --------------------------------------------------------
    # No drift
    # --------------------------------------------------------

    if not drift_found:

        print(
            "\nNo drift detected."
        )

        log_event(
            "retraining_skipped",
            {
                "reason": "No data drift detected"
            }
        )

        return {
            "drift_detected": False,
            "retraining_triggered": False,
            "message": "No drift detected"
        }

    # --------------------------------------------------------
    # Drift detected
    # --------------------------------------------------------

    print(
        "\nDrift detected."
    )

    print(
        "Starting automated retraining..."
    )

    log_event(
        "retraining_started",
        {
            "reason": "Data drift detected"
        }
    )

    try:

        run_pipeline(
            auto_deploy=True
        )

        log_event(
            "retraining_completed",
            {
                "reason": "Data drift detected"
            }
        )

        return {
            "drift_detected": True,
            "retraining_triggered": True,
            "message": "Retraining and deployment completed"
        }

    except Exception as e:

        log_event(
            "retraining_failed",
            {
                "error": str(e)
            },
            status="failed"
        )

        print(
            f"\nRetraining failed: {e}"
        )

        return {
            "drift_detected": True,
            "retraining_triggered": True,
            "message": "Retraining failed",
            "error": str(e)
        }
# ============================================================
# MAIN AUTOMATED ML PIPELINE
# ============================================================

def run_pipeline(
    dataset_path=None,
    target_column=None,
    auto_deploy=True
):
    """
    Execute the complete AutoMLOps training pipeline.

    Steps:
    1. Prepare dataset
    2. Train multiple models
    3. Select best model
    4. Register model in MLflow
    5. Optionally deploy the registered version
    """

    # --------------------------------------------------------
    # Use existing Iris workflow by default
    # --------------------------------------------------------

    if dataset_path is None:

        load_and_save_data()

        dataset_path = DEFAULT_DATASET

    if target_column is None:

        target_column = DEFAULT_TARGET

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    results = train_models(
        data_path=dataset_path,
        target_column=target_column
    )

    # --------------------------------------------------------
    # Select and register best model
    # --------------------------------------------------------

    select_and_save_best_model(
        results,
        target_column=target_column,
        dataset_path=dataset_path
    )

    # --------------------------------------------------------
    # Deploy registered model
    # --------------------------------------------------------

    if auto_deploy:

        from deploy_model import deploy_model

        deployment = deploy_model()

        print(
            f"\nModel deployed successfully:"
        )

        print(
            f"Model: "
            f"{deployment['model_name']}"
        )

        print(
            f"Version: "
            f"{deployment['model_version']}"
        )

    print(
        "\nPipeline execution completed."
    )