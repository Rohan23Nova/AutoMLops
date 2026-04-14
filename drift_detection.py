import pandas as pd
from scipy.stats import ks_2samp
from monitoring import log_event


def detect_drift(reference_df, current_df, threshold=0.05):
    drift_report = {}

    for column in reference_df.columns:
        stat, p_value = ks_2samp(reference_df[column], current_df[column])

        drift_report[column] = {
            "p_value": p_value,
            "drift_detected": p_value < threshold
        }

    return drift_report