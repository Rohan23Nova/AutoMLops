from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

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

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0]),
        "class_name": label_map[int(prediction[0])]
    }