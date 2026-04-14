import json
from datetime import datetime

LOG_FILE = "logs/monitoring_log.json"

def log_event(event_type, details=None, status="success"):
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "event": event_type,
        "status": status,
        "details": details
    }

    with open("logs/monitoring_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)
        
def log_prediction(input_data, prediction, mode="single"):
    entry = {
        "timestamp": str(datetime.datetime.now()),
        "mode": mode,
        "input": input_data,
        "prediction": prediction
    }

    with open("logs/prediction_history.json", "a") as f:
        f.write(json.dumps(entry) + "\n")