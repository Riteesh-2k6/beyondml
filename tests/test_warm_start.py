"""
Tests for Genetic Algorithm Warm-Starting.
"""

import pytest
from beyondml.engine.genetic import Genome, GeneticModelOptimizer

def test_genome_warm_start():
    """Verify that a Genome with warm_start=True receives standard hyperparameters."""
    # RandomForest warm start
    genome_rf = Genome(problem_type="classification", num_features=10, model_choice="RandomForest", warm_start=True)
    assert genome_rf.hparams["n_estimators"] == 100
    assert genome_rf.hparams["max_depth"] is None
    
    # LogisticRegression warm start
    genome_lr = Genome(problem_type="classification", num_features=10, model_choice="LogisticRegression", warm_start=True)
    assert genome_lr.hparams["C"] == 1.0
    assert genome_lr.hparams["max_iter"] == 1000

def test_optimizer_warm_start_injection(classification_profile, classification_df):
    """Verify that the first Genome in the optimizer's population is warm-started."""
    optimizer = GeneticModelOptimizer(
        df=classification_df,
        target_column="target",
        profile=classification_profile,
        pop_size=5,
        model_choice="RandomForest"
    )
    
    # The first genome should have the default warm start params
    first_genome = optimizer.population[0]
    assert getattr(first_genome, "warm_start", False) is True
    assert first_genome.hparams["n_estimators"] == 100
    
    # Subsequent genomes should NOT be warm-started (unless by pure random chance they generate the exact same dict,
    # but their `warm_start` flag should be False).
    second_genome = optimizer.population[1]
    assert getattr(second_genome, "warm_start", False) is False
