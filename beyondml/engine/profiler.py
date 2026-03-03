"""
Dataset Profiler & Target Identifier — deterministic analysis layer.
Ported from data/profiling/profiler.py with minor cleanup.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import re


class TargetIdentifier:
    """Heuristic-based target column identification."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def identify(self) -> Dict[str, Any]:
        scores = {}
        for col in self.df.columns:
            score = 0
            series = self.df[col]

            if series.nunique() <= 1:
                continue
            missing_ratio = series.isnull().mean()
            if missing_ratio > 0.8:
                continue

            unique_ratio = series.nunique() / len(series)
            col_lower = col.lower()

            if re.search(r"target|label|class|outcome|y|result|status|pred|dependent|species|category", col_lower):
                score += 5
            if re.search(r"id|uuid|index|key|name|serial|date|time|timestamp", col_lower):
                score -= 4

            num_unique = series.nunique()
            if series.dtype == "object" or str(series.dtype) == "category":
                if 2 <= num_unique <= 50:
                    score += 3
                elif num_unique > 50:
                    score -= 2
            elif pd.api.types.is_numeric_dtype(series):
                if 2 <= num_unique <= 20:
                    score += 2
                elif num_unique > 20:
                    score += 1
                if series.std() == 0:
                    score -= 5

            col_idx = list(self.df.columns).index(col)
            if col_idx == 0 or col_idx == len(self.df.columns) - 1:
                score += 1

            score -= missing_ratio * 2
            if unique_ratio > 0.9:
                score -= 3

            scores[col] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return {"suggested_target": None, "confidence_score": 0, "ranked_candidates": []}

        best_col, best_score = ranked[0]
        confidence = min(max(best_score / 8, 0), 1)
        return {
            "suggested_target": best_col,
            "confidence_score": float(confidence),
            "ranked_candidates": ranked,
        }


class DatasetProfiler:
    """Comprehensive dataset profiling for ML pipeline decisions."""

    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df.copy()
        self.target_column = target_column
        self.feature_columns = [c for c in df.columns if c != target_column]

    def run(self) -> Dict[str, Any]:
        return {
            "metadata": self._basic_metadata(),
            "feature_types": self._infer_feature_types(),
            "missing_analysis": self._missing_analysis(),
            "target_analysis": self._target_analysis() if self.target_column else None,
            "numerical_summary": self._numerical_summary(),
            "correlation_matrix": self._correlation_matrix(),
            "correlation_summary": self._correlation_summary(),
            "categorical_cardinality": self._categorical_cardinality(),
            "outlier_summary": self._outlier_summary(),
            "overfitting_risk_index": self._calculate_ori(),
        }

    def _calculate_ori(self) -> Dict[str, Any]:
        """
        Compute Overfitting Risk Index (ORI).
        ORI = w1*r + w2*I + w3*(1-MI) + w4*Corr + w5*Noise
        """
        n = len(self.df)
        d = len(self.feature_columns)
        
        # 1. Dimensional Ratio (r = D/N) - Normalized to [0,1]
        r = min(d / n, 1.0) if n > 0 else 1.0

        # 2. Imbalance Ratio (I) - Normalized [0,1]
        target_analysis = self._target_analysis()
        imbalance = 0.0
        if target_analysis and target_analysis.get("target_type") == "classification":
            # imb_ratio is min class pct. If perfectly balanced (0.5 for binary), I=0. If rare (0.01), I=0.99
            imb_ratio = target_analysis.get("imbalance_ratio", 0.5)
            # Normalize: 0.5 -> 0, 0.0 -> 1
            imbalance = 1.0 - (imb_ratio * 2.0) if imb_ratio < 0.5 else 0.0

        # 3. Mutual Information (MI)
        mi_val = self._average_mutual_information()
        one_minus_mi = 1.0 - mi_val

        # 4. Feature Correlation (Corr)
        corr_summary = self._correlation_summary()
        # Avg correlation of numeric columns
        numerics = self.df.select_dtypes(include=[np.number])
        if numerics.shape[1] > 1:
            corr_mean = numerics.corr().abs().values[np.triu_indices(numerics.shape[1], k=1)].mean()
            corr_mean = 0.0 if np.isnan(corr_mean) else corr_mean
        else:
            corr_mean = 0.0

        # 5. Noise Estimate
        # Proxy: Percentage of data points that are outliers
        outliers = self._outlier_summary()
        noise_pct = sum(v["pct"] for v in outliers.values()) / max(len(outliers), 1)

        # Weights w1..w5 (Sum = 1)
        w = [0.3, 0.2, 0.2, 0.2, 0.1]
        ori = (w[0] * r + w[1] * imbalance + w[2] * one_minus_mi + w[3] * corr_mean + w[4] * noise_pct)
        ori = float(min(max(ori, 0), 1))

        return {
            "score": round(ori, 4),
            "metrics": {
                "d_n_ratio": round(r, 4),
                "imbalance": round(imbalance, 4),
                "one_minus_mi": round(one_minus_mi, 4),
                "mean_corr": round(corr_mean, 4),
                "noise_proxy": round(noise_pct, 4)
            }
        }

    def _average_mutual_information(self) -> float:
        """Heuristic for Mutual Information using correlation as a proxy for this layer."""
        if not self.target_column: return 0.5
        # For a prompt analysis, we'll use a simplified correlation-based MI proxy
        # to avoid bringing in sklearn.feature_selection here if not needed.
        numerics = self.df.select_dtypes(include=[np.number])
        if self.target_column in numerics.columns:
            target_corr = numerics.corr()[self.target_column].abs().drop(self.target_column).mean()
            return float(min(target_corr if not np.isnan(target_corr) else 0.3, 1.0))
        return 0.4 # Default mid-point for categorical/complex target mi proxy

    def _basic_metadata(self) -> Dict[str, Any]:
        return {
            "num_rows": int(self.df.shape[0]),
            "num_columns": int(self.df.shape[1]),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / (1024**2)),
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
            "missing_percentage": {k: float(v) for k, v in missing_pct.items()},
            "high_missing_cols": [c for c, p in missing_pct.items() if p > 0.4],
        }

    def _target_analysis(self) -> Dict[str, Any]:
        if not self.target_column or self.target_column not in self.df.columns:
            return None
        target = self.df[self.target_column]
        nu = target.nunique()

        if not pd.api.types.is_numeric_dtype(target) or nu < 15:
            problem_type = "classification"
        else:
            problem_type = "regression"

        analysis = {"target_type": problem_type, "num_unique": int(nu)}
        if problem_type == "classification":
            dist = target.value_counts(normalize=True).to_dict()
            analysis["class_distribution"] = {str(k): float(v) for k, v in dist.items()}
            analysis["imbalance_ratio"] = float(min(dist.values()))
        else:
            analysis["mean"] = float(target.mean())
            analysis["std"] = float(target.std())
            analysis["skewness"] = float(target.skew())
        return analysis

    def _numerical_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        summary = {}
        for col in numerics.columns:
            if col == self.target_column:
                continue
            s = numerics[col]
            summary[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
                "median": round(float(s.median()), 4),
            }
        return summary

    def _correlation_matrix(self) -> Dict[str, Any]:
        """Full correlation matrix as nested dict for chart rendering."""
        numerics = self.df.select_dtypes(include=[np.number])
        feature_numerics = numerics[[c for c in numerics.columns if c != self.target_column]]
        if feature_numerics.shape[1] < 2:
            return {}
        corr = feature_numerics.corr()
        return {col: {row: round(float(corr.loc[row, col]), 2) for row in corr.index} for col in corr.columns}

    def _correlation_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        if numerics.shape[1] < 2:
            return {"high_correlation_pairs": []}
        corr = numerics.corr().abs()
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] > 0.8:
                    pairs.append({"f1": cols[i], "f2": cols[j], "val": round(float(corr.iloc[i, j]), 4)})
        return {"high_correlation_pairs": pairs}

    def _categorical_cardinality(self) -> Dict[str, Any]:
        cats = self.df.select_dtypes(exclude=[np.number])
        return {col: int(self.df[col].nunique()) for col in cats.columns if col != self.target_column}

    def _outlier_summary(self) -> Dict[str, Any]:
        numerics = self.df.select_dtypes(include=[np.number])
        outliers = {}
        for col in numerics.columns:
            if col == self.target_column:
                continue
            s = numerics[col].dropna()
            if len(s) == 0:
                continue
            mean, std = s.mean(), s.std()
            if std == 0:
                continue
            z_scores = ((s - mean) / std).abs()
            outlier_mask = z_scores > 2.5
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outliers[col] = {
                    "count": outlier_count,
                    "pct": round(float(outlier_count / len(s)), 4),
                    "indices": s[outlier_mask].index.tolist()[:20],  # cap at 20
                }
        return outliers
