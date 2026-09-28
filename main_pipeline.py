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

    if drift_found:

        print(
            "\nDrift detected. "
            "Retraining model..."
        )

        run_pipeline()

    else:

        print(
            "\nNo drift detected."
        )


# ============================================================
# MAIN AUTOMATED ML PIPELINE
# ============================================================

def run_pipeline(
    dataset_path=None,
    target_column=None
):
    """
    Execute the complete AutoMLOps pipeline.

    Parameters
    ----------
    dataset_path : str, optional
        Path to the dataset.

    target_column : str, optional
        Name of the target column.

    If no values are provided, the existing Iris
    pipeline is used for backward compatibility.
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
    # Select best model
    # --------------------------------------------------------

    select_and_save_best_model(
        results,
        target_column=target_column,
        dataset_path=dataset_path
    )

    print(
        "\nPipeline execution completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()