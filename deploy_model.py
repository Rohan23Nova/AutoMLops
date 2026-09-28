import json
import os
import mlflow


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# ============================================================
# PATHS
# ============================================================

METADATA_PATH = "models/model_metadata.json"
DEPLOYMENT_PATH = "models/deployment.json"


# ============================================================
# PROMOTE MODEL
# ============================================================

def deploy_model(version=None):
    """
    Promote a registered MLflow model version for serving.

    If no version is supplied, the latest registered version
    from model_metadata.json is used.
    """

    # --------------------------------------------------------
    # Load training metadata
    # --------------------------------------------------------

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r"
    ) as f:

        metadata = json.load(f)

    model_name = metadata[
        "mlflow_model_name"
    ]

    if version is None:

        version = metadata[
            "mlflow_model_version"
        ]

    if version is None:
        raise ValueError(
            "No MLflow model version available."
        )

    # --------------------------------------------------------
    # Verify model exists in MLflow
    # --------------------------------------------------------

    client = mlflow.MlflowClient()

    try:

        registered_model = client.get_model_version(
            name=model_name,
            version=str(version)
        )

    except Exception as e:

        raise ValueError(
            f"Could not find model "
            f"{model_name} version {version}: {e}"
        )

    # --------------------------------------------------------
    # Create deployment record
    # --------------------------------------------------------

    deployment = {
        "model_name": model_name,
        "model_version": str(version),
        "status": "deployed",
        "source_run_id": registered_model.run_id,
        "deployment_timestamp": __import__(
            "datetime"
        ).datetime.now().isoformat()
    }

    os.makedirs(
        "models",
        exist_ok=True
    )

    with open(
        DEPLOYMENT_PATH,
        "w"
    ) as f:

        json.dump(
            deployment,
            f,
            indent=4
        )

    print("\n========== MODEL DEPLOYMENT ==========")
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print("Status: deployed")
    print(f"Deployment record: {DEPLOYMENT_PATH}")

    return deployment


# ============================================================
# COMMAND LINE ENTRY
# ============================================================

if __name__ == "__main__":

    deploy_model()