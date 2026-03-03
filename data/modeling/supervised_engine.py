# supervised_engine.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, List, Tuple

from evaluation.metrics import calculate_metrics

class SupervisedPipeline:
    def __init__(self, df: pd.DataFrame, target_column: str, profile: Dict[str, Any]):
        self.df = df
        self.target_column = target_column
        self.profile = profile
        self.problem_type = profile['target_analysis']['target_type']
        
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        
    def run_baselines(self) -> Dict[str, Any]:
        """
        Executes baseline models and returns performance metrics.
        """
        # Split data
        stratify = self.y if self.problem_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Build Preprocessor
        preprocessor = self._build_preprocessor()
        
        results = {}
        
        # Define models
        if self.problem_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_val)
            y_pred_train = pipeline.predict(X_train)
            
            # Calculate metrics
            val_metrics = calculate_metrics(y_val, y_pred, self.problem_type)
            train_metrics = calculate_metrics(y_train, y_pred_train, self.problem_type)
            
            results[name] = {
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "overfitting_gap": self._calc_gap(train_metrics, val_metrics)
            }
            
        return results

    def _build_preprocessor(self) -> ColumnTransformer:
        num_cols = self.profile['feature_types']['numerical']
        cat_cols = self.profile['feature_types']['categorical']
        
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ],
            remainder='drop' # Drop columns like text or datetime for now
        )
        
        return preprocessor

    def _calc_gap(self, train: Dict[str, float], val: Dict[str, float]) -> float:
        if self.problem_type == "classification":
            return train.get("accuracy", 0) - val.get("accuracy", 0)
        else:
            return val.get("rmse", 0) - train.get("rmse", 0) # For error metrics, gap is inverted
