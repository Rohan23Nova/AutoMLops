import pickle
import os
import json
from datetime import datetime

def select_and_save_best_model(results):
    # Make sure models folder exists
    os.makedirs("models", exist_ok=True)

    # Select best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = results[best_model_name]["model"]
    best_f1 = results[best_model_name]["f1"]

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nBest Model Selected:", best_model_name)
    print("Best F1 Score:", results[best_model_name]["f1"])
    
    print("Model saved to models/best_model.pkl")
    metadata = {
    "model_name": best_model_name,
    "f1_score": best_f1,
    "timestamp": str(datetime.now())
}

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)