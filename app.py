from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm
)

from pydantic import BaseModel

from jose import JWTError, jwt

import json
import os
import logging
import threading

import pandas as pd
import mlflow
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
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
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="AutoMLOps API",
    description="Automated Machine Learning Operations API",
    version="2.0"
)


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
# MLFLOW
# ============================================================

mlflow.set_experiment(
    "AutoMLOps_Inference"
)



# ============================================================
# LOAD REGISTERED MODEL FROM MLFLOW
# ============================================================



# ============================================================
# LOAD DEPLOYED MODEL INFORMATION
# ============================================================

DEPLOYMENT_PATH = "models/deployment.json"

if not os.path.exists(DEPLOYMENT_PATH):

    raise FileNotFoundError(
        "No deployed model found. "
        "Run deploy_model.py first."
    )

with open(
    DEPLOYMENT_PATH,
    "r"
) as f:

    deployment = json.load(f)


MODEL_NAME = deployment[
    "model_name"
]

MODEL_VERSION = deployment[
    "model_version"
]

MODEL_STATUS = deployment[
    "status"
]

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

model = mlflow.sklearn.load_model(
    MODEL_URI
)

print("Registered model loaded successfully.")


# ============================================================
# AUTHENTICATION
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
        "message": "AutoMLOps API is running"
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
# GENERIC SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    data: PredictionInput,
    user: str = Depends(get_current_user)
):

    try:

        # --------------------------------------------------------
        # Convert incoming dictionary to DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame(
            [data.data]
        )

        # --------------------------------------------------------
        # Run complete saved pipeline
        # --------------------------------------------------------

        prediction = model.predict(df)

        pred_class = prediction[0]

        # Convert NumPy types into normal Python types
        if hasattr(pred_class, "item"):
            pred_class = pred_class.item()

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
            "prediction": pred_class
        }

    except Exception as e:

        logging.error(
            f"Prediction error: {str(e)}"
        )

        return {
            "error": str(e)
        }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/batch_predict")
def batch_predict(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):

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
        # Run complete pipeline
        # --------------------------------------------------------

        predictions = model.predict(df)

        predictions_list = predictions.tolist()

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
            "predictions": predictions_list
        }

    except Exception as e:

        logging.error(
            f"Batch prediction error: {str(e)}"
        )

        return {
            "error": str(e)
        }


# ============================================================
# RETRAIN
# ============================================================

@app.post("/retrain")
def retrain(
    user: str = Depends(get_current_user)
):

    thread = threading.Thread(
        target=run_pipeline
    )

    thread.start()

    return {
        "message": "Retraining started in background"
    }


# ============================================================
# DRIFT CHECK
# ============================================================

@app.post("/check-drift")
def check_drift():

    from main_pipeline import check_and_retrain

    check_and_retrain()

    return {
        "message": "Drift check completed"
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