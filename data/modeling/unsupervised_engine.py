# unsupervised_engine.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List

class UnsupervisedPipeline:
    def __init__(self, df: pd.DataFrame, profile: Dict[str, Any]):
        self.df = df
        self.profile = profile
        self.numeric_cols = profile['feature_types']['numerical']
        
    def run_clustering(self) -> Dict[str, Any]:
        """
        Runs clustering algorithms and returns metrics.
        """
        # Prepare data (Numerical only for now)
        X = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # 1. KMeans
        for k in [2, 3, 5]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            sil = float(silhouette_score(X_scaled, labels))
            results[f"KMeans (k={k})"] = {
                "silhouette_score": sil,
                "inertia": float(kmeans.inertia_)
            }
            
        # 2. DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        # Silhouette score only works if more than 1 cluster and no noise (-1)
        unique_labels = set(labels)
        if len(unique_labels) > 1 and (len(unique_labels) > 2 or -1 not in unique_labels):
            sil = float(silhouette_score(X_scaled, labels))
        else:
            sil = 0.0
            
        results["DBSCAN"] = {
            "silhouette_score": sil,
            "num_clusters": int(len(unique_labels) - (1 if -1 in unique_labels else 0)),
            "noise_points": int(list(labels).count(-1))
        }
        
        # 3. PCA for Variance Structure
        pca = PCA(n_components=min(len(self.numeric_cols), 3))
        pca.fit(X_scaled)
        results["PCA"] = {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "total_variance_explained": float(np.sum(pca.explained_variance_ratio_))
        }
        
        return results
