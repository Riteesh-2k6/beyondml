"""
Supervised ML Pipeline — baseline models + full training.
Ported from data/modeling/supervised_engine.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Tuple

from .metrics import calculate_metrics


class SupervisedPipeline:
    def __init__(self, df: pd.DataFrame, target_column: str, profile: Dict[str, Any]):
        self.df = df
        self.target_column = target_column
        self.profile = profile
        self.problem_type = profile["target_analysis"]["target_type"]
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    def run_baselines(self) -> Dict[str, Any]:
        stratify = self.y if self.problem_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify
        )
        preprocessor = self._build_preprocessor()
        results = {}

        if self.problem_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            }
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            }

        for name, model in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_pred_train = pipe.predict(X_train)
            val_metrics = calculate_metrics(y_val, y_pred, self.problem_type)
            train_metrics = calculate_metrics(y_train, y_pred_train, self.problem_type)
            results[name] = {
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "overfitting_gap": self._calc_gap(train_metrics, val_metrics),
            }
        return results

    def train_final_model(self, model_obj, X_train=None, y_train=None, X_test=None, y_test=None):
        """Train a final model and return (pipeline, train_metrics, test_metrics, feature_importances)."""
        if X_train is None:
            stratify = self.y if self.problem_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=stratify
            )
        preprocessor = self._build_preprocessor()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model_obj)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_pred_train = pipe.predict(X_train)
        test_metrics = calculate_metrics(y_test, y_pred, self.problem_type)
        train_metrics = calculate_metrics(y_train, y_pred_train, self.problem_type)

        # Feature importances
        importances = {}
        try:
            if hasattr(model_obj, "feature_importances_"):
                fi = model_obj.feature_importances_
                feature_names = self._get_feature_names(preprocessor, X_train)
                for name, imp in zip(feature_names, fi):
                    importances[name] = round(float(imp), 4)
        except Exception:
            pass
        return pipe, train_metrics, test_metrics, importances

    def _get_feature_names(self, preprocessor, X):
        """Extract feature names from fitted preprocessor."""
        try:
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            return [f"feature_{i}" for i in range(X.shape[1])]

    def _build_preprocessor(self) -> ColumnTransformer:
        num_cols = [c for c in self.profile["feature_types"]["numerical"] if c in self.X.columns]
        cat_cols = [c for c in self.profile["feature_types"]["categorical"] if c in self.X.columns]
        num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer(
            transformers=[("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)],
            remainder="drop",
        )

    def _calc_gap(self, train: Dict, val: Dict) -> float:
        if self.problem_type == "classification":
            return round(train.get("accuracy", 0) - val.get("accuracy", 0), 4)
        return round(val.get("rmse", 0) - train.get("rmse", 0), 4)
