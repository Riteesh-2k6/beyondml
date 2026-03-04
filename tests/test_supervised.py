"""Tests for SupervisedPipeline — baselines and final model training."""

import pytest
from beyondml.engine.supervised import SupervisedPipeline


class TestSupervisedBaselines:
    def test_baselines_returns_models(self, classification_df, classification_profile):
        pipeline = SupervisedPipeline(classification_df, "target", classification_profile)
        results = pipeline.run_baselines()

        assert len(results) >= 2
        for name, res in results.items():
            assert "val_metrics" in res
            assert "train_metrics" in res
            assert "overfitting_gap" in res

    def test_classification_metrics_present(self, classification_df, classification_profile):
        pipeline = SupervisedPipeline(classification_df, "target", classification_profile)
        results = pipeline.run_baselines()

        for name, res in results.items():
            vm = res["val_metrics"]
            assert "accuracy" in vm
            assert 0.0 <= vm["accuracy"] <= 1.0

    def test_regression_baselines(self, regression_df, regression_profile):
        pipeline = SupervisedPipeline(regression_df, "target", regression_profile)
        results = pipeline.run_baselines()

        assert len(results) >= 2
        for name, res in results.items():
            vm = res["val_metrics"]
            assert "rmse" in vm
            assert "r2" in vm


class TestFinalModel:
    def test_train_final_model(self, classification_df, classification_profile):
        from sklearn.ensemble import RandomForestClassifier

        pipeline = SupervisedPipeline(classification_df, "target", classification_profile)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        pipe, train_m, test_m, importances = pipeline.train_final_model(model)

        assert pipe is not None
        assert "accuracy" in train_m
        assert "accuracy" in test_m
        assert isinstance(importances, dict)
