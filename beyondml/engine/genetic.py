"""
Genetic Algorithm Model Optimizer with callback support for TUI.
Ported from data/modeling/genetic_algorithm.py with gen_callback added.
"""

import numpy as np
import random
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

from .metrics import calculate_metrics


def _evaluate_genome_worker(
    genome: "Genome",
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    numeric_features: List[str],
    problem_type: str,
    profile: Dict[str, Any],
) -> Tuple[float, Dict]:
    """Standalone worker function for parallel evaluation to avoid pickling issues."""
    selected = [f for i, f in enumerate(feature_names) if genome.feature_mask[i] == 1]
    numeric_selected = [f for f in selected if f in numeric_features]
    if not numeric_selected:
        return -1.0, {}

    X_sub = X[numeric_selected]

    try:
        if genome.model_choice == "RandomForest":
            model = (
                RandomForestClassifier(**genome.hparams, random_state=42)
                if problem_type == "classification"
                else RandomForestRegressor(**genome.hparams, random_state=42)
            )
        elif genome.model_choice == "LogisticRegression":
            model = LogisticRegression(**genome.hparams)
        elif genome.model_choice == "SVM":
            model = SVC(**genome.hparams) if problem_type == "classification" else SVR(**genome.hparams)
        elif genome.model_choice == "DecisionTree":
            model = (
                DecisionTreeClassifier(**genome.hparams, random_state=42)
                if problem_type == "classification"
                else DecisionTreeRegressor(**genome.hparams, random_state=42)
            )
        elif genome.model_choice == "KNN":
            model = (
                KNeighborsClassifier(**genome.hparams)
                if problem_type == "classification"
                else KNeighborsRegressor(**genome.hparams)
            )
        elif genome.model_choice == "GradientBoosting":
            model = (
                GradientBoostingClassifier(**genome.hparams, random_state=42)
                if problem_type == "classification"
                else GradientBoostingRegressor(**genome.hparams, random_state=42)
            )
        else:
            model = LinearRegression()

        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

        # Perform K-Fold CV
        cv_scores = cross_val_score(
            pipe, X_sub, y, cv=5, scoring="accuracy" if problem_type == "classification" else "r2"
        )

        mu_cv = np.mean(cv_scores)
        sigma_cv = np.std(cv_scores)

        # Generalization Gap
        pipe.fit(X_sub, y)
        train_pred = pipe.predict(X_sub)
        train_metrics = calculate_metrics(y, train_pred, problem_type)
        mu_train = train_metrics.get("accuracy" if problem_type == "classification" else "r2", 0)
        generalization_gap = max(0, mu_train - mu_cv)

        # Complexity Penalty C(P)
        feat_penalty = len(selected) / len(feature_names)
        model_penalty = 0.0
        if genome.model_choice in ("RandomForest", "GradientBoosting"):
            n_est = genome.hparams.get("n_estimators", 100)
            depth = genome.hparams.get("max_depth", 10) or 20
            model_penalty = (n_est / 250) * 0.5 + (max(2, depth) / 20) * 0.5
        elif genome.model_choice == "DecisionTree":
            depth = genome.hparams.get("max_depth", 10) or 20
            model_penalty = max(2, depth) / 20

        c_p = (feat_penalty * 0.4) + (model_penalty * 0.6)

        # Adaptive Lambda scaling from ORI
        ori_score = profile.get("overfitting_risk_index", {}).get("score", 0.5)
        beta = 1.5
        l1 = 0.05 * (1 + beta * ori_score)
        l2 = 0.15 * (1 + beta * ori_score)
        l3 = 0.10 * (1 + beta * ori_score)

        fitness = mu_cv - (l1 * c_p) - (l2 * sigma_cv) - (l3 * generalization_gap)
        metrics = {"mu_cv": mu_cv, "sigma_cv": sigma_cv, "gap": generalization_gap, "complexity": c_p}
        return fitness, metrics

    except Exception:
        return -1.0, {}


class Genome:
    """Encodes a model configuration: model_type, hyperparameters, feature_mask."""

    def __init__(self, problem_type: str, num_features: int, model_choice: str = None):
        self.problem_type = problem_type
        self.num_features = num_features

        if model_choice:
            self.model_choice = model_choice
        elif problem_type == "classification":
            self.model_choice = random.choice(["RandomForest", "LogisticRegression"])
        else:
            self.model_choice = random.choice(["RandomForest", "LinearRegression"])

        self.hparams = self._init_hparams()
        self.feature_mask = [random.choice([0, 1]) for _ in range(num_features)]
        if sum(self.feature_mask) == 0:
            self.feature_mask[random.randint(0, num_features - 1)] = 1

        self.fitness = 0.0
        self.metrics = {}

    def _init_hparams(self) -> Dict[str, Any]:
        if self.model_choice == "RandomForest":
            return {
                "n_estimators": random.choice([50, 100, 150, 200, 250]),
                "max_depth": random.choice([None, 2, 5, 7, 10, 15, 20]),
                "min_samples_split": random.randint(2, 10),
                "min_samples_leaf": random.randint(1, 5),
                "max_features": random.choice(["sqrt", "log2", 0.3, 0.5, 0.7]),
            }
        elif self.model_choice == "LogisticRegression":
            return {"C": random.uniform(0.01, 10.0), "max_iter": 1000}
        elif self.model_choice == "LinearRegression":
            return {}
        elif self.model_choice == "SVM":
            return {"C": random.uniform(0.1, 10.0), "kernel": random.choice(["rbf", "linear", "poly"])}
        elif self.model_choice == "DecisionTree":
            return {
                "max_depth": random.choice([None, 3, 5, 10, 15, 20]),
                "min_samples_split": random.randint(2, 10),
            }
        elif self.model_choice == "KNN":
            return {"n_neighbors": random.choice([3, 5, 7, 9, 11, 15])}
        elif self.model_choice == "GradientBoosting":
            return {
                "n_estimators": random.choice([50, 100, 150, 200]),
                "learning_rate": random.choice([0.01, 0.05, 0.1, 0.2]),
                "max_depth": random.choice([3, 5, 7]),
            }
        return {}


class GeneticModelOptimizer:
    """GA optimizer with callback-based progress reporting for TUI integration."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        profile: Dict[str, Any],
        pop_size: int = 10,
        generations: int = 5,
        model_choice: str = None,
        gen_callback: Optional[Callable] = None,
    ):
        self.df = df
        self.target_column = target_column
        self.profile = profile
        self.pop_size = pop_size
        self.generations = generations
        self.model_choice = model_choice
        self.gen_callback = gen_callback
        self.problem_type = profile["target_analysis"]["target_type"]

        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.feature_names = self.X.columns.tolist()
        self.numeric_features = [c for c in self.feature_names if pd.api.types.is_numeric_dtype(df[c])]

        self.population = [
            Genome(self.problem_type, len(self.feature_names), model_choice=self.model_choice)
            for _ in range(pop_size)
        ]

    def evolve(self) -> Tuple[List[Dict], "Genome"]:
        history = []

        for gen in range(self.generations):
            # Parallelize genome evaluation using standalone worker to avoid pickling 'self' (Spec 2.0 optimization)
            results = Parallel(n_jobs=-1)(
                delayed(_evaluate_genome_worker)(
                    genome,
                    self.X,
                    self.y,
                    self.feature_names,
                    self.numeric_features,
                    self.problem_type,
                    self.profile,
                )
                for genome in self.population
                if genome.fitness == 0 or gen == 0
            )
            
            # Map results back to population
            eval_idx = 0
            for genome in self.population:
                if genome.fitness == 0 or gen == 0:
                    fitness, metrics = results[eval_idx]
                    genome.fitness = fitness
                    genome.metrics = metrics
                    eval_idx += 1

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]

            gen_summary = {
                "gen": gen + 1,
                "best_fitness": round(best.fitness, 4),
                "avg_fitness": round(np.mean([g.fitness for g in self.population]), 4),
                "best_model": best.model_choice,
                "num_features": sum(best.feature_mask),
                "best_hparams": best.hparams.copy(),
            }
            history.append(gen_summary)

            # Fire callback for TUI updates
            if self.gen_callback:
                self.gen_callback(gen_summary)

            # Spec 7.3: Selection & reproduction
            # Elitism fraction 0.05 <= elite_ratio <= 0.15
            elite_count = max(1, int(self.pop_size * 0.10))
            new_pop = self.population[:elite_count] 

            # Diversity preservation / Selection
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_pop.extend([c1, c2])
            self.population = new_pop[: self.pop_size]
            
            # Spec 7.7: Early stopping
            if gen > 2:
                last_best = history[-2]["best_fitness"]
                if abs(gen_summary["best_fitness"] - last_best) < 1e-4:
                    # simplistic patience - for real patience we'd track a counter
                    pass 
        
        # Spec 8.0: Final selection protocol - find best candidate
        best = max(self.population, key=lambda x: x.fitness)
        return history, best

    def _evaluate(self, genome: Genome):
        """Sequential evaluation fallback."""
        fitness, metrics = _evaluate_genome_worker(
            genome,
            self.X,
            self.y,
            self.feature_names,
            self.numeric_features,
            self.problem_type,
            self.profile,
        )
        genome.fitness = fitness
        genome.metrics = metrics

    def _tournament_select(self) -> Genome:
        # Spec 7.1: T must satisfy 2 <= T <= population/5
        t_size = max(2, min(3, self.pop_size // 5))
        contestants = random.sample(self.population, t_size)
        return max(contestants, key=lambda x: x.fitness)

    def _crossover(self, p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
        cp = random.randint(0, len(p1.feature_mask) - 1)
        c1 = Genome(self.problem_type, len(self.feature_names), model_choice=self.model_choice or p1.model_choice)
        c2 = Genome(self.problem_type, len(self.feature_names), model_choice=self.model_choice or p2.model_choice)
        c1.feature_mask = p1.feature_mask[:cp] + p2.feature_mask[cp:]
        c2.feature_mask = p2.feature_mask[:cp] + p1.feature_mask[cp:]
        c1.hparams = p1.hparams.copy()
        c2.hparams = p2.hparams.copy()
        return c1, c2

    def _mutate(self, genome: Genome):
        # Spec 7.2: Mutation rate floor m >= 0.05
        m_rate = 0.15 # Higher than floor for diversity
        for i in range(len(genome.feature_mask)):
            if random.random() < m_rate:
                genome.feature_mask[i] = 1 - genome.feature_mask[i]
        
        if sum(genome.feature_mask) == 0:
            genome.feature_mask[random.randint(0, len(genome.feature_mask) - 1)] = 1

        if genome.model_choice == "RandomForest":
            if random.random() < 0.2:
                genome.hparams["n_estimators"] = max(10, genome.hparams.get("n_estimators", 100) + random.randint(-30, 30))
            if random.random() < 0.2:
                genome.hparams["max_depth"] = random.choice([None, 2, 5, 7, 10, 15, 20])
        elif genome.model_choice == "LogisticRegression":
            if random.random() < 0.2:
                genome.hparams["C"] = max(0.001, genome.hparams.get("C", 1.0) * random.uniform(0.5, 1.5))
