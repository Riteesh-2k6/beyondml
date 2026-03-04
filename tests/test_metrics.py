"""Tests for calculate_metrics utility."""

import pytest
import numpy as np
from beyondml.engine.metrics import calculate_metrics


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 1, 1, 0]
        m = calculate_metrics(y_true, y_pred, "classification")

        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["precision_macro"] == 1.0
        assert m["recall_macro"] == 1.0

    def test_imperfect_predictions(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 0, 1]
        m = calculate_metrics(y_true, y_pred, "classification")

        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0

    def test_all_keys_present(self):
        m = calculate_metrics([1, 0], [1, 0], "classification")
        for key in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
            assert key in m


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        m = calculate_metrics(y_true, y_pred, "regression")

        assert m["mse"] == 0.0
        assert m["rmse"] == 0.0
        assert m["mae"] == 0.0
        assert m["r2"] == 1.0

    def test_imperfect_predictions(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.5, 2.5, 2.5, 3.5]
        m = calculate_metrics(y_true, y_pred, "regression")

        assert m["mse"] >= 0
        assert m["rmse"] >= 0
        assert m["mae"] >= 0
        assert m["r2"] <= 1.0

    def test_all_keys_present(self):
        m = calculate_metrics([1.0], [1.0], "regression")
        for key in ["mse", "rmse", "mae", "r2"]:
            assert key in m
