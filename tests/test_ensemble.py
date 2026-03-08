"""
Tests for the Ensemble Engine and Ensemble Agent.
"""
import pytest
import json
import pandas as pd
import numpy as np
from beyondml.engine.ensemble import EnsembleEngine, genome_to_estimator
from beyondml.agents.ensemble_agent import EnsembleAgent


class FakeGenome:
    """Lightweight genome mock for ensemble testing."""
    def __init__(self, model_choice, hparams=None, fitness=0.8):
        self.model_choice = model_choice
        self.hparams = hparams or {}
        self.fitness = fitness
        self.feature_mask = [1, 1, 1, 1]
        self.metrics = {"mu_cv": fitness}


def test_genome_to_estimator_random_forest():
    g = FakeGenome("RandomForest", {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"})
    est = genome_to_estimator(g, "classification")
    assert hasattr(est, "fit")
    assert hasattr(est, "predict")


def test_genome_to_estimator_logistic_regression():
    g = FakeGenome("LogisticRegression", {"C": 1.0, "max_iter": 1000})
    est = genome_to_estimator(g, "classification")
    assert hasattr(est, "fit")


def test_ensemble_stacking_build():
    genomes = [
        FakeGenome("RandomForest", {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}, 0.9),
        FakeGenome("LogisticRegression", {"C": 1.0, "max_iter": 1000}, 0.85),
        FakeGenome("DecisionTree", {"max_depth": 5, "min_samples_split": 2}, 0.80),
    ]
    engine = EnsembleEngine("classification")
    stacker = engine.build_stacking(genomes)
    assert hasattr(stacker, "fit")
    assert hasattr(stacker, "predict")
    assert len(stacker.estimators) == 3


def test_ensemble_voting_build():
    genomes = [
        FakeGenome("RandomForest", {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}, 0.9),
        FakeGenome("GradientBoosting", {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}, 0.85),
    ]
    engine = EnsembleEngine("classification")
    voter = engine.build_voting(genomes)
    assert hasattr(voter, "fit")
    assert len(voter.estimators) == 2


def test_ensemble_stacking_train_and_predict(classification_df):
    """End-to-end: build stacking ensemble, train, and predict."""
    genomes = [
        FakeGenome("RandomForest", {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}, 0.9),
        FakeGenome("LogisticRegression", {"C": 1.0, "max_iter": 1000}, 0.85),
    ]
    engine = EnsembleEngine("classification")
    stacker = engine.build_stacking(genomes)

    X = classification_df.drop(columns=["target", "category"])
    y = classification_df["target"]
    stacker.fit(X, y)
    preds = stacker.predict(X)
    assert len(preds) == len(y)


@pytest.mark.asyncio
async def test_ensemble_agent_integration(classification_df, mock_llm):
    """Integration: EnsembleAgent trains a stacking ensemble from fake genomes."""
    genomes = [
        FakeGenome("RandomForest", {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}, 0.9),
        FakeGenome("LogisticRegression", {"C": 1.0, "max_iter": 1000}, 0.85),
    ]

    from beyondml.engine.profiler import DatasetProfiler
    profiler = DatasetProfiler(classification_df, target_column="target")
    profile = profiler.run()

    async def mock_log(msg): pass

    agent = EnsembleAgent(mock_llm)
    result = await agent.run(
        df=classification_df,
        target_column="target",
        profile=profile,
        top_genomes=genomes,
        problem_type="classification",
        log=mock_log,
    )

    assert "test_score" in result
    assert result["strategy"] == "stacking"
    assert result["test_score"] >= 0
    assert len(result["base_models"]) == 2
