import json
import logging
import os
import threading
import time

import mlflow
import pandas as pd

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm
)

from pydantic import BaseModel

from jose import JWTError, jwt

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest
)

from main_pipeline import run_pipeline

from auth import (
    create_access_token,
    verify_user,
    SECRET_KEY,
    ALGORITHM
)

from monitoring import (
    log_event,
    log_prediction
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "AutoMLOps_Inference"
)


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="AutoMLOps API",
    description="Automated Machine Learning Operations API",
    version="2.1"
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

prediction_counter = Counter(
    "automlops_predictions_total",
    "Total number of successful single predictions"
)

prediction_error_counter = Counter(
    "automlops_prediction_errors_total",
    "Total number of prediction errors"
)

batch_prediction_counter = Counter(
    "automlops_batch_predictions_total",
    "Total number of successful batch prediction requests"
)

prediction_latency = Histogram(
    "automlops_prediction_latency_seconds",
    "Prediction request latency in seconds"
)

batch_prediction_latency = Histogram(
    "automlops_batch_prediction_latency_seconds",
    "Batch prediction request latency in seconds"
)

model_version_gauge = Gauge(
    "automlops_model_version",
    "Currently deployed model version"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="login"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename="logs/api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# ============================================================
# LOAD DEPLOYED MODEL INFORMATION
# ============================================================

DEPLOYMENT_PATH = "models/deployment.json"


if not os.path.exists(DEPLOYMENT_PATH):
    raise FileNotFoundError(
        "No deployed model found. "
        "Run deploy_model.py first."
    )


with open(DEPLOYMENT_PATH, "r") as f:
    deployment = json.load(f)


MODEL_NAME = deployment["model_name"]
MODEL_VERSION = deployment["model_version"]
MODEL_STATUS = deployment["status"]


if MODEL_STATUS != "deployed":
    raise RuntimeError(
        f"Model status is '{MODEL_STATUS}', "
        "not 'deployed'."
    )


MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


print("\n========== MODEL LOADING ==========")
print(f"Model: {MODEL_NAME}")
print(f"Version: {MODEL_VERSION}")
print(f"URI: {MODEL_URI}")


# ============================================================
# LOAD MODEL FROM MLFLOW REGISTRY
# ============================================================

model = mlflow.sklearn.load_model(
    MODEL_URI
)

print("Registered model loaded successfully.")


# Set Prometheus gauge
try:
    model_version_gauge.set(float(MODEL_VERSION))
except (TypeError, ValueError):
    model_version_gauge.set(0)


# ============================================================
# AUTHENTICATION FUNCTION
# ============================================================

def get_current_user(
    token: str = Depends(oauth2_scheme)
):
    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        return username

    except JWTError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


# ============================================================
# GENERIC PREDICTION INPUT
# ============================================================

class PredictionInput(BaseModel):
    """
    Generic prediction input.

    Any number of feature fields can be supplied.
    Values may be numerical or categorical.
    """

    data: dict


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AutoMLOps API is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": MODEL_STATUS
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):

    user = verify_user(
        form_data.username,
        form_data.password
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    access_token = create_access_token(
        data={
            "sub": user["username"]
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    data: PredictionInput,
    user: str = Depends(get_current_user)
):

    start_time = time.time()

    try:

        # --------------------------------------------------------
        # Convert input dictionary to DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame(
            [data.data]
        )

        # --------------------------------------------------------
        # Run deployed model
        # --------------------------------------------------------

        prediction = model.predict(df)

        pred_class = prediction[0]

        # Convert NumPy types into normal Python types
        if hasattr(pred_class, "item"):
            pred_class = pred_class.item()

        # --------------------------------------------------------
        # Prometheus metrics
        # --------------------------------------------------------

        prediction_counter.inc()

        prediction_latency.observe(
            time.time() - start_time
        )

        # --------------------------------------------------------
        # Log prediction
        # --------------------------------------------------------

        log_prediction(
            input_data=data.data,
            prediction=pred_class,
            mode="single"
        )

        logging.info(
            f"User: {user} | "
            f"Input: {data.data} | "
            f"Prediction: {pred_class}"
        )

        return {
            "prediction": pred_class,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

    except Exception as e:

        prediction_error_counter.inc()

        logging.error(
            f"Prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/batch_predict")
def batch_predict(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):

    start_time = time.time()

    try:

        # --------------------------------------------------------
        # Read CSV
        # --------------------------------------------------------

        df = pd.read_csv(
            file.file
        )

        # --------------------------------------------------------
        # Validate dataset
        # --------------------------------------------------------

        if df.empty:

            raise ValueError(
                "Uploaded CSV is empty."
            )

        # --------------------------------------------------------
        # Run deployed model
        # --------------------------------------------------------

        predictions = model.predict(df)

        predictions_list = predictions.tolist()

        # --------------------------------------------------------
        # Prometheus metrics
        # --------------------------------------------------------

        batch_prediction_counter.inc()

        batch_prediction_latency.observe(
            time.time() - start_time
        )

        # --------------------------------------------------------
        # Log prediction
        # --------------------------------------------------------

        log_prediction(
            input_data={
                "filename": file.filename,
                "rows": len(df)
            },
            prediction=predictions_list,
            mode="batch"
        )

        log_event(
            event_type="batch_prediction",
            details={
                "filename": file.filename,
                "rows": len(df)
            },
            status="success"
        )

        logging.info(
            f"User: {user} | "
            f"Batch file: {file.filename} | "
            f"Rows: {len(df)}"
        )

        return {
            "rows_received": len(df),
            "predictions": predictions_list,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

    except Exception as e:

        prediction_error_counter.inc()

        logging.error(
            f"Batch prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# RETRAIN
# ============================================================

@app.post("/retrain")
def retrain(
    user: str = Depends(get_current_user)
):

    def retrain_task():

        try:

            run_pipeline(
                dataset_path="data/processed/current.csv",
                target_column="target",
                auto_deploy=True
            )

            log_event(
                event_type="manual_retraining",
                details={
                    "triggered_by": user
                },
                status="success"
            )

        except Exception as e:

            logging.error(
                f"Retraining error: {str(e)}"
            )

            log_event(
                event_type="manual_retraining",
                details={
                    "triggered_by": user,
                    "error": str(e)
                },
                status="failed"
            )


    thread = threading.Thread(
        target=retrain_task
    )

    thread.start()

    return {
        "message": "Retraining started in background"
    }


# ============================================================
# DRIFT CHECK
# ============================================================

@app.post("/check-drift")
def check_drift(
    user: str = Depends(get_current_user)
):

    from main_pipeline import check_and_retrain

    result = check_and_retrain(
        auto_deploy=True
    )

    return {
        "message": "Drift check completed",
        "result": result
    }


# ============================================================
# LOGS
# ============================================================

@app.get("/logs")
def get_logs(
    user: str = Depends(get_current_user)
):

    try:

        with open(
            "logs/monitoring_log.json",
            "r"
        ) as f:

            data = json.load(f)

        return data

    except Exception:

        return []


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status": MODEL_STATUS,
        "source_run_id": deployment.get(
            "source_run_id"
        ),
        "deployment_timestamp": deployment.get(
            "deployment_timestamp"
        )
    }


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )