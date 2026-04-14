from fastapi import FastAPI
import json
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import mlflow
from main_pipeline import run_pipeline
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from auth import create_access_token, verify_user
from jose import JWTError, jwt
from auth import SECRET_KEY, ALGORITHM
from pydantic import BaseModel
from fastapi import UploadFile, File
import pandas as pd
import pickle
from monitoring import log_event
from monitoring import log_prediction



class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

mlflow.set_experiment("AutoMLOps_Inference")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

logging.basicConfig(
    filename="logs/api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
app = FastAPI()
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
# Load model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Label mapping
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.get("/")
def home():
    return {"message": "AutoMLOps API is running"}

@app.post("/predict")
def predict(data: InputData, user: str = Depends(get_current_user)):
    try:
        import pandas as pd

        df = pd.DataFrame([data.dict()])

        df.columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]

        prediction = model.predict(df)
        pred_class = int(prediction[0])

        return {
            "prediction": pred_class
        }

    except Exception as e:
        return {
            "error": str(e)
        }
@app.post("/retrain")
def retrain(user: str = Depends(get_current_user)):
    thread = threading.Thread(target=run_pipeline)
    thread.start()

    return {"message": "Retraining started in background"}
@app.post("/check-drift")
def check_drift():
    from main_pipeline import check_and_retrain

    check_and_retrain()
    return {"message": "Drift check completed"}
@app.get("/logs")
def get_logs(user: str = Depends(get_current_user)):
    try:
        with open("logs/monitoring_log.json", "r") as f:
            data = json.load(f)
        return data
    except:
        return []
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = verify_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# load model (if not already global)
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/batch_predict")
def batch_predict(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    log_prediction(
    input_data="batch_file",
    prediction=predictions.tolist(),
    mode="batch"
)
    try:
        df = pd.read_csv(file.file)

        predictions = model.predict(df)

        return {
            "rows_received": len(df),
            "predictions": predictions.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
    
    log_event(
        event_type="batch_prediction",
        details={"rows": len(df)},
        status="success"
    )