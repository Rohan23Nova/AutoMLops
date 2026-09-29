import pandas as pd
from scipy.stats import ks_2samp
from monitoring import log_event


def detect_drift(reference_df, current_df, threshold=0.05):
    drift_report = {}

    numerical_columns = reference_df.select_dtypes(
        include=["number"]
    ).columns

    for column in numerical_columns:

        if column not in current_df.columns:
            continue

        reference_values = reference_df[column].dropna()
        current_values = current_df[column].dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            continue

        stat, p_value = ks_2samp(
            reference_values,
            current_values
        )

        drift_report[column] = {
            "test": "Kolmogorov-Smirnov",
            "p_value": float(p_value),
            "drift_detected": bool(p_value < threshold)
        }

    drift_detected = any(
        result["drift_detected"]
        for result in drift_report.values()
    )

    log_event(
        "drift_check",
        {
            "threshold": threshold,
            "drift_detected": drift_detected,
            "report": drift_report
        }
    )

    return drift_report