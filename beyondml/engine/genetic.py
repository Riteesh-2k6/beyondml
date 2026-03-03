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

from .metrics import calculate_metrics


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
            for genome in self.population:
                self._evaluate(genome)

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

            # Selection & reproduction
            new_pop = self.population[:2]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_pop.extend([c1, c2])
            self.population = new_pop[: self.pop_size]

        # Return best overall
        best = max(self.population, key=lambda x: x.fitness)
        return history, best

    def _evaluate(self, genome: Genome):
        selected = [f for i, f in enumerate(self.feature_names) if genome.feature_mask[i] == 1]
        numeric_selected = [f for f in selected if f in self.numeric_features]
        if not numeric_selected:
            genome.fitness = 0.0
            return

        X_sub = self.X[numeric_selected]
        X_train, X_val, y_train, y_val = train_test_split(X_sub, self.y, test_size=0.2, random_state=42)

        if genome.model_choice == "RandomForest":
            model = (
                RandomForestClassifier(**genome.hparams, random_state=42)
                if self.problem_type == "classification"
                else RandomForestRegressor(**genome.hparams, random_state=42)
            )
        elif genome.model_choice == "LogisticRegression":
            model = LogisticRegression(**genome.hparams)
        elif genome.model_choice == "SVM":
            model = SVC(**genome.hparams) if self.problem_type == "classification" else SVR(**genome.hparams)
        elif genome.model_choice == "DecisionTree":
            model = (
                DecisionTreeClassifier(**genome.hparams, random_state=42)
                if self.problem_type == "classification"
                else DecisionTreeRegressor(**genome.hparams, random_state=42)
            )
        elif genome.model_choice == "KNN":
            model = (
                KNeighborsClassifier(**genome.hparams)
                if self.problem_type == "classification"
                else KNeighborsRegressor(**genome.hparams)
            )
        elif genome.model_choice == "GradientBoosting":
            model = (
                GradientBoostingClassifier(**genome.hparams, random_state=42)
                if self.problem_type == "classification"
                else GradientBoostingRegressor(**genome.hparams, random_state=42)
            )
        else:
            model = LinearRegression()

        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred, self.problem_type)

            if self.problem_type == "classification":
                perf = metrics.get("accuracy", 0)
            else:
                perf = max(metrics.get("r2", 0), 0)

            complexity_penalty = 0.1 * (1.0 - (sum(genome.feature_mask) / len(self.feature_names)))
            genome.fitness = perf + complexity_penalty
            genome.metrics = metrics
        except Exception:
            genome.fitness = 0.0

    def _tournament_select(self) -> Genome:
        contestants = random.sample(self.population, min(3, len(self.population)))
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
        for i in range(len(genome.feature_mask)):
            if random.random() < 0.1:
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
