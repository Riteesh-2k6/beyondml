"""Unsupervised ML Pipeline — clustering and PCA."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, Any


class UnsupervisedPipeline:
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.df = df
        self.profile = profile
        self.numeric_cols = [c for c in profile["feature_types"]["numerical"] if c in df.columns]

    def run_clustering(self) -> Dict[str, Any]:
        X = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        results = {}

        for k in [2, 3, 5]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels))
            results[f"KMeans (k={k})"] = {"silhouette_score": round(sil, 4), "inertia": round(float(kmeans.inertia_), 2)}

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        unique_labels = set(labels)
        if len(unique_labels) > 1 and (len(unique_labels) > 2 or -1 not in unique_labels):
            sil = float(silhouette_score(X_scaled, labels))
        else:
            sil = 0.0
        results["DBSCAN"] = {
            "silhouette_score": round(sil, 4),
            "num_clusters": int(len(unique_labels) - (1 if -1 in unique_labels else 0)),
            "noise_points": int(list(labels).count(-1)),
        }

        pca = PCA(n_components=min(len(self.numeric_cols), 3))
        pca.fit(X_scaled)
        results["PCA"] = {
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_variance_explained": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        }
        return results
