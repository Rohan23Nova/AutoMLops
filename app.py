from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import logging
import mlflow
from main_pipeline import run_pipeline
import threading
from fastapi.middleware.cors import CORSMiddleware


mlflow.set_experiment("AutoMLOps_Inference")

logging.basicConfig(
    filename="logs/api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def predict(data: IrisInput):
    features = np.array([
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]).reshape(1, -1)

    with mlflow.start_run():
        prediction = model.predict(features)
        pred_class = int(prediction[0])

        mlflow.log_param("input", str(data.dict()))
        mlflow.log_metric("prediction", pred_class)

    logging.info(f"Input: {data.dict()} | Prediction: {pred_class}")

    return {
        "prediction": pred_class,
        "class_name": label_map[pred_class]
    }
@app.post("/retrain")
def retrain():
    thread = threading.Thread(target=run_pipeline)
    thread.start()

    return {"message": "Retraining started in background"}