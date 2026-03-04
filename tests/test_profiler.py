"""Tests for DatasetProfiler, TargetIdentifier, and ORI computation."""

import pytest
from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier


class TestTargetIdentifier:
    def test_finds_a_target(self, classification_df):
        identifier = TargetIdentifier(classification_df)
        result = identifier.identify()

        assert "suggested_target" in result
        assert "confidence_score" in result
        assert "ranked_candidates" in result
        assert result["suggested_target"] is not None

    def test_confidence_in_range(self, classification_df):
        identifier = TargetIdentifier(classification_df)
        result = identifier.identify()

        assert 0.0 <= result["confidence_score"] <= 1.0


class TestDatasetProfiler:
    EXPECTED_KEYS = [
        "metadata",
        "feature_types",
        "missing_analysis",
        "target_analysis",
        "numerical_summary",
        "correlation_summary",
        "categorical_cardinality",
        "outlier_summary",
        "overfitting_risk_index",
    ]

    def test_returns_all_keys(self, classification_df):
        profiler = DatasetProfiler(classification_df, target_column="target")
        profile = profiler.run()

        for key in self.EXPECTED_KEYS:
            assert key in profile, f"Missing key: {key}"

    def test_feature_types_detected(self, classification_df):
        profiler = DatasetProfiler(classification_df, target_column="target")
        profile = profiler.run()

        ft = profile["feature_types"]
        assert "numerical" in ft
        assert "categorical" in ft
        assert len(ft["numerical"]) > 0

    def test_ori_score_in_range(self, classification_df):
        profiler = DatasetProfiler(classification_df, target_column="target")
        profile = profiler.run()

        ori = profile.get("overfitting_risk_index", {})
        assert "score" in ori
        assert 0.0 <= ori["score"] <= 1.0

    def test_regression_profile(self, regression_df):
        profiler = DatasetProfiler(regression_df, target_column="target")
        profile = profiler.run()

        assert profile["target_analysis"]["target_type"] in ("regression", "classification")
        assert profile["metadata"]["num_rows"] == 200
