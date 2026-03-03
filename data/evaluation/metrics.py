# metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from typing import Dict, Any

def calculate_metrics(y_true, y_pred, problem_type: str) -> Dict[str, float]:
    """
    Calculates multi-objective metrics based on problem type.
    """
    metrics = {}
    
    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        # Use macro F1 to account for imbalance
        try:
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro'))
        except:
            metrics["f1_macro"] = 0.0
            
        # ROC-AUC requires probability scores or decision function, but here we use predictions for simplicity
        # In a real GA, we might want AUC. For baseline, we keep it simple.
    else:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        
    return metrics
