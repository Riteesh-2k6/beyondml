"""Tests for UnsupervisedPipeline — clustering and PCA."""

import pytest
import numpy as np
from beyondml.engine.unsupervised import UnsupervisedPipeline


class TestUnsupervisedPipeline:
    def test_clustering_returns_results(self, classification_df, classification_profile):
        pipeline = UnsupervisedPipeline(classification_df, classification_profile)
        results = pipeline.run_clustering()

        assert "KMeans (k=2)" in results
        assert "KMeans (k=3)" in results
        assert "DBSCAN" in results
        assert "PCA" in results

    def test_silhouette_scores_valid(self, classification_df, classification_profile):
        pipeline = UnsupervisedPipeline(classification_df, classification_profile)
        results = pipeline.run_clustering()

        for key in ["KMeans (k=2)", "KMeans (k=3)", "KMeans (k=5)"]:
            sil = results[key]["silhouette_score"]
            assert -1.0 <= sil <= 1.0

    def test_pca_variance_sums_to_one_or_less(self, classification_df, classification_profile):
        pipeline = UnsupervisedPipeline(classification_df, classification_profile)
        results = pipeline.run_clustering()

        pca = results["PCA"]
        total = pca["total_variance_explained"]
        assert 0.0 < total <= 1.0001  # Small float tolerance
