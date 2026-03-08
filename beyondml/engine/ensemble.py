"""
Ensemble Engine — Stacking and Voting strategies for combining top-N GA genomes.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def genome_to_estimator(genome, problem_type: str):
    """Convert a GA Genome into a fitted sklearn estimator."""
    mc = genome.model_choice
    hp = genome.hparams.copy()
    # Remove max_iter from hp — we handle it explicitly per model
    hp.pop("max_iter", None)

    if mc == "RandomForest":
        cls = RandomForestClassifier if problem_type == "classification" else RandomForestRegressor
        hp["random_state"] = 42
        return cls(**hp)
    elif mc == "LogisticRegression":
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=1000)
    elif mc == "LinearRegression":
        return LinearRegression()
    elif mc == "SVM":
        cls = SVC if problem_type == "classification" else SVR
        hp.setdefault("probability", True)
        return cls(**hp)
    elif mc == "DecisionTree":
        cls = DecisionTreeClassifier if problem_type == "classification" else DecisionTreeRegressor
        return cls(**hp)
    elif mc == "KNN":
        cls = KNeighborsClassifier if problem_type == "classification" else KNeighborsRegressor
        return cls(**hp)
    elif mc == "GradientBoosting":
        cls = GradientBoostingClassifier if problem_type == "classification" else GradientBoostingRegressor
        hp["random_state"] = 42
        return cls(**hp)
    else:
        cls = RandomForestClassifier if problem_type == "classification" else RandomForestRegressor
        return cls(random_state=42)


def _deduplicate_genomes(genomes) -> list:
    """Keep only genomes with distinct model types for ensemble diversity."""
    seen = set()
    unique = []
    for g in genomes:
        key = g.model_choice
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


class EnsembleEngine:
    """Builds stacking or voting ensembles from top-N GA genomes."""

    def __init__(self, problem_type: str):
        self.problem_type = problem_type

    def build_stacking(self, genomes, meta_learner=None):
        """Build a StackingClassifier/Regressor from top-N genomes."""
        unique = _deduplicate_genomes(genomes)
        if len(unique) < 2:
            # Need at least 2 diverse models to stack — duplicate with different params
            unique = genomes[:2] if len(genomes) >= 2 else genomes

        estimators = [
            (f"{g.model_choice}_{i}", genome_to_estimator(g, self.problem_type))
            for i, g in enumerate(unique)
        ]

        if meta_learner is None:
            if self.problem_type == "classification":
                meta_learner = LogisticRegression(max_iter=1000)
            else:
                meta_learner = LinearRegression()

        if self.problem_type == "classification":
            return StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False,
            )
        else:
            return StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False,
            )

    def build_voting(self, genomes, voting="soft"):
        """Build a VotingClassifier/Regressor from top-N genomes."""
        unique = _deduplicate_genomes(genomes)
        if len(unique) < 2:
            unique = genomes[:2] if len(genomes) >= 2 else genomes

        estimators = [
            (f"{g.model_choice}_{i}", genome_to_estimator(g, self.problem_type))
            for i, g in enumerate(unique)
        ]

        if self.problem_type == "classification":
            return VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=-1,
            )
        else:
            return VotingRegressor(
                estimators=estimators,
                n_jobs=-1,
            )
