import json
import os
import math
from datetime import datetime

LOG_FILE = "logs/monitoring_log.json"
PREDICTION_LOG_FILE = "logs/prediction_history.json"


def ensure_logs_directory():
    os.makedirs("logs", exist_ok=True)


def make_json_safe(obj):
    """
    Convert NumPy/Pandas values and non-finite numbers
    into values that are valid JSON.
    """

    # NumPy scalar values
    if hasattr(obj, "item"):
        try:
            return make_json_safe(obj.item())
        except Exception:
            pass

    # Dictionaries
    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # Float NaN / Infinity
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None

    # JSON-compatible primitive values
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # Fallback
    return str(obj)


def load_json_array(file_path):
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
    safe_data = make_json_safe(data)

    with open(file_path, "w") as f:
        json.dump(
            safe_data,
            f,
            indent=4,
            allow_nan=False
        )


def log_event(event_type, details=None, status="success"):
    ensure_logs_directory()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "status": status,
        "details": make_json_safe(details)
    }

    data = load_json_array(LOG_FILE)

    data.append(log_entry)

    save_json_array(LOG_FILE, data)


def log_prediction(input_data, prediction, mode="single"):
    ensure_logs_directory()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "input": make_json_safe(input_data),
        "prediction": make_json_safe(prediction)
    }

    data = load_json_array(PREDICTION_LOG_FILE)

    data.append(entry)

    save_json_array(PREDICTION_LOG_FILE, data)