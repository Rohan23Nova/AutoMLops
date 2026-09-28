import json
import os
from datetime import datetime


LOG_FILE = "logs/monitoring_log.json"
PREDICTION_LOG_FILE = "logs/prediction_history.json"


def ensure_logs_directory():
    os.makedirs("logs", exist_ok=True)


def load_json_array(file_path):
    """
    Load a JSON array from a file.
    Returns [] if the file does not exist or is invalid.
    """

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_json_array(file_path, data):
    """
    Save a Python list as formatted JSON.
    """

    with open(file_path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


def log_event(
    event_type,
    details=None,
    status="success"
):
    """
    Log an AutoMLOps system event.
    """

    ensure_logs_directory()

    log_entry = {
        "timestamp": str(datetime.now()),
        "event": event_type,
        "status": status,
        "details": details
    }

    data = load_json_array(
        LOG_FILE
    )

    data.append(log_entry)

    save_json_array(
        LOG_FILE,
        data
    )


def log_prediction(
    input_data,
    prediction,
    mode="single"
):
    """
    Log a model prediction.
    """

    ensure_logs_directory()

    entry = {
        "timestamp": str(datetime.now()),
        "mode": mode,
        "input": input_data,
        "prediction": prediction
    }

    data = load_json_array(
        PREDICTION_LOG_FILE
    )

    data.append(entry)

    save_json_array(
        PREDICTION_LOG_FILE,
        data
    )