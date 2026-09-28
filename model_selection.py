import pickle
import os
import json
from datetime import datetime
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def select_and_save_best_model(
    results,
    target_column="target",
    dataset_path="data/processed/iris.csv"
):
    """
    Select the best model based on F1 score,
    save it locally, and register it in MLflow.
    """

    os.makedirs("models", exist_ok=True)

    # ============================================================
    # 1. SELECT BEST MODEL
    # ============================================================

    best_model_name = max(
        results,
        key=lambda x: results[x]["f1"]
    )

    best_model = results[
        best_model_name
    ]["model"]

    best_f1 = results[
        best_model_name
    ]["f1"]

    best_run_id = results[
        best_model_name
    ]["run_id"]

    # ============================================================
    # 2. SAVE LOCAL MODEL
    # ============================================================

    model_path = "models/best_model.pkl"

    with open(
        model_path,
        "wb"
    ) as f:

        pickle.dump(
            best_model,
            f
        )

    print(
        f"\nBest Model Selected: "
        f"{best_model_name}"
    )

    print(
        f"Best F1 Score: "
        f"{best_f1:.4f}"
    )

    print(
        f"Model saved to {model_path}"
    )

    # ============================================================
    # 3. REGISTER MODEL IN MLFLOW
    # ============================================================

    model_name = "AutoMLOps_Model"

    model_uri = (
        f"runs:/{best_run_id}/model"
    )

    try:

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        model_version = registered_model.version

        print(
            f"\nMLflow Model Registered:"
        )

        print(
            f"Model Name: {model_name}"
        )

        print(
            f"Version: {model_version}"
        )

    except Exception as e:

        print(
            f"\nMLflow model registration failed: "
            f"{e}"
        )

        model_version = None

    # ============================================================
    # 4. SAVE METADATA
    # ============================================================

    metadata = {
        "model_name": best_model_name,
        "f1_score": best_f1,
        "target_column": target_column,
        "dataset_path": dataset_path,
        "mlflow_run_id": best_run_id,
        "mlflow_model_name": model_name,
        "mlflow_model_version": model_version,
        "timestamp": str(datetime.now()),
        "pipeline_type": "preprocessing + model"
    }

    metadata_path = (
        "models/model_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Metadata saved to {metadata_path}"
    )

    return best_model