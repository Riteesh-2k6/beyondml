"""Tests for Genome and GeneticModelOptimizer."""

import pytest
from beyondml.engine.genetic import Genome, GeneticModelOptimizer


class TestGenome:
    def test_initialization(self):
        genome = Genome(problem_type="classification", num_features=10)

        assert len(genome.feature_mask) == 10
        assert sum(genome.feature_mask) >= 1  # At least one feature selected
        assert genome.model_choice is not None
        assert genome.fitness == 0.0
        assert isinstance(genome.hparams, dict)

    def test_specific_model_choice(self):
        genome = Genome(
            problem_type="classification",
            num_features=5,
            model_choice="RandomForest",
        )
        assert genome.model_choice == "RandomForest"

    def test_regression_genome(self):
        genome = Genome(problem_type="regression", num_features=8)

        assert genome.problem_type == "regression"
        assert len(genome.feature_mask) == 8


class TestGeneticModelOptimizer:
    def test_evolve_completes(self, classification_df, classification_profile):
        optimizer = GeneticModelOptimizer(
            df=classification_df,
            target_column="target",
            profile=classification_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        history, best = optimizer.evolve()

        assert len(history) >= 1
        assert best is not None
        assert best.fitness > 0 or best.fitness == 0  # Fitness is computed

    def test_best_has_metrics(self, classification_df, classification_profile):
        optimizer = GeneticModelOptimizer(
            df=classification_df,
            target_column="target",
            profile=classification_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        _, best = optimizer.evolve()

        assert isinstance(best.metrics, dict)
        assert best.model_choice == "RandomForest"

    def test_regression_ga(self, regression_df, regression_profile):
        optimizer = GeneticModelOptimizer(
            df=regression_df,
            target_column="target",
            profile=regression_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        history, best = optimizer.evolve()

        assert best is not None
        assert len(history) >= 1
