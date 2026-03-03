# profiler.py

import pandas as pd
import numpy as np
from typing import Dict, Any
import re

class TargetIdentifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def identify(self):
        scores = {}

        for col in self.df.columns:
            score = 0
            series = self.df[col]

            # 1. Skip constant or all-missing columns
            if series.nunique() <= 1:
                continue
            
            missing_ratio = series.isnull().mean()
            if missing_ratio > 0.8:
                continue

            unique_ratio = series.nunique() / len(series)

            # 2. Name Semantics (Strong Signal)
            col_lower = col.lower()
            if re.search(r"target|label|class|outcome|y|result|status|pred|dependent", col_lower):
                score += 5
            if re.search(r"id|uuid|index|key|name|serial|date|time", col_lower):
                score -= 4

            # 3. Cardinality & Distribution
            num_unique = series.nunique()
            
            # Classification Target heuristics
            if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
                if 2 <= num_unique <= 50:
                    score += 3
                elif num_unique > 50:
                    score -= 2 # High cardinality categorical is less likely to be target
            
            # Regression Target heuristics
            elif pd.api.types.is_numeric_dtype(series):
                if 2 <= num_unique <= 20: # Likely discrete target
                    score += 2
                elif num_unique > 20: # Potential continuous target
                    score += 1
                
                # Distribution check: Target varies significantly
                if series.std() == 0:
                    score -= 5

            # 4. Position Heuristic (Often target is first or last)
            col_idx = list(self.df.columns).index(col)
            if col_idx == 0 or col_idx == len(self.df.columns) - 1:
                score += 1

            # 5. Missing Values penalty
            score -= (missing_ratio * 2)

            # 6. Uniqueness Penalty (IDs are usually unique)
            if unique_ratio > 0.9:
                score -= 3

            scores[col] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            return {"suggested_target": None, "confidence_score": 0, "ranked_candidates": []}

        best_col, best_score = ranked[0]
        # Normalize confidence (roughly)
        confidence = min(max(best_score / 8, 0), 1)

        return {
            "suggested_target": best_col,
            "confidence_score": float(confidence),
            "ranked_candidates": ranked
        }


class DatasetProfiler:
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]

    def run(self) -> Dict[str, Any]:
        return {
            "metadata": self._basic_metadata(),
            "feature_types": self._infer_feature_types(),
            "missing_analysis": self._missing_analysis(),
            "target_analysis": self._target_analysis() if self.target_column else None,
            "numerical_summary": self._numerical_summary(),
            "correlation_summary": self._correlation_summary(),
            "categorical_cardinality": self._categorical_cardinality(),
            "outlier_summary": self._outlier_summary()
        }

    def _basic_metadata(self) -> Dict[str, Any]:
        return {
            "num_rows": int(self.df.shape[0]),
            "num_columns": int(self.df.shape[1]),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / (1024 ** 2))
        }

    def _infer_feature_types(self) -> Dict[str, Any]:
        types = {"numerical": [], "categorical": [], "boolean": [], "datetime": [], "text": []}
        
        for col in self.feature_columns:
            series = self.df[col]
            if pd.api.types.is_bool_dtype(series):
                types["boolean"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                types["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(series):
                types["numerical"].append(col)
            else:
                nu = series.nunique()
                if nu < 20 or (nu / len(series) < 0.1):
                    types["categorical"].append(col)
                else:
                    types["text"].append(col)
        return types

    def _missing_analysis(self) -> Dict[str, Any]:
        missing_pct = self.df.isnull().mean().to_dict()
        return {
            "missing_percentage": missing_pct,
            "high_missing_cols": [c for c, p in missing_pct.items() if p > 0.4]
        }

    def _target_analysis(self) -> Dict[str, Any]:
        if not self.target_column or self.target_column not in self.df.columns:
            return None
            
        target_series = self.df[self.target_column]
        nu = target_series.nunique()
        
        if not pd.api.types.is_numeric_dtype(target_series) or nu < 15:
            problem_type = "classification"
        else:
            problem_type = "regression"
            
        analysis = {"target_type": problem_type, "num_unique": int(nu)}
        
        if problem_type == "classification":
            dist = target_series.value_counts(normalize=True).to_dict()
            analysis["class_distribution"] = dist
            analysis["imbalance_ratio"] = float(min(dist.values()))
        else:
            analysis["mean"] = float(target_series.mean())
            analysis["std"] = float(target_series.std())
            analysis["skewness"] = float(target_series.skew())
            
        return analysis

    def _numerical_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        summary = {}
        for col in numerics.columns:
            if col == self.target_column: continue
            s = numerics[col]
            summary[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "range": float(s.max() - s.min()),
                "skewness": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "median": float(s.median())
            }
        return summary

    def _correlation_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        if numerics.shape[1] < 2: return {"high_correlation_pairs": []}
        
        corr = numerics.corr().abs()
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if corr.iloc[i, j] > 0.8:
                    pairs.append({"f1": cols[i], "f2": cols[j], "val": float(corr.iloc[i, j])})
        return {"high_correlation_pairs": pairs}

    def _categorical_cardinality(self) -> Dict[str, Any]:
        cats = self.df.select_dtypes(exclude=[np.number])
        return {col: int(self.df[col].nunique()) for col in cats.columns if col != self.target_column}

    def _outlier_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        outliers = {}
        for col in numerics.columns:
            if col == self.target_column: continue
            s = numerics[col].dropna()
            if len(s) == 0: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            pct = len(s[(s < lower) | (s > upper)]) / len(s)
            outliers[col] = {"outlier_pct": float(pct)}
        return outliers
