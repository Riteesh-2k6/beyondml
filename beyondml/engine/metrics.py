"""Metric calculations for classification and regression."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from typing import Dict


def calculate_metrics(y_true, y_pred, problem_type: str) -> Dict[str, float]:
    metrics = {}
    if problem_type == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        try:
            metrics["f1_macro"] = round(float(f1_score(y_true, y_pred, average="macro")), 4)
            metrics["precision_macro"] = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            metrics["recall_macro"] = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        except Exception:
            metrics["f1_macro"] = 0.0
    else:
        metrics["mse"] = round(float(mean_squared_error(y_true, y_pred)), 4)
        metrics["rmse"] = round(float(np.sqrt(metrics["mse"])), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)
        metrics["r2"] = round(float(r2_score(y_true, y_pred)), 4)
    return metrics
