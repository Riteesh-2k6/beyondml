# genetic_algorithm.py

import numpy as np
import random
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from modeling.supervised_engine import SupervisedPipeline
from evaluation.metrics import calculate_metrics

class Genome:
    """
    Encodes a model configuration: model_type, hyperparameters, and feature_mask.
    """
    def __init__(self, problem_type: str, num_features: int, initial_params: Dict[str, Any] = None):
        self.problem_type = problem_type
        self.num_features = num_features
        
        # 1. Model Type
        if problem_type == "classification":
            self.model_choice = random.choice(["RandomForest", "LogisticRegression"])
        else:
            self.model_choice = random.choice(["RandomForest", "LinearRegression"])
            
        # 2. Hyperparameters (Simplified)
        self.hparams = initial_params or self._init_hparams()
        
        # 3. Feature Mask (Binary vector)
        self.feature_mask = [random.choice([0, 1]) for _ in range(num_features)]
        if sum(self.feature_mask) == 0: # Ensure at least one feature
            self.feature_mask[random.randint(0, num_features-1)] = 1
            
        self.fitness = 0.0
        self.metrics = {}

    def _init_hparams(self) -> Dict[str, Any]:
        if self.model_choice == "RandomForest":
            return {
                "n_estimators": random.randint(10, 200),
                "max_depth": random.choice([None, 5, 10, 20]),
                "min_samples_split": random.randint(2, 10)
            }
        elif self.model_choice == "LogisticRegression":
            return {"C": random.uniform(0.01, 10.0)}
        elif self.model_choice == "LinearRegression":
            return {} # Basic
        return {}

class GeneticModelOptimizer:
    def __init__(self, df: pd.DataFrame, target_column: str, profile: Dict[str, Any], pop_size=10, generations=5):
        self.df = df
        self.target_column = target_column
        self.profile = profile
        self.pop_size = pop_size
        self.generations = generations
        self.problem_type = profile['target_analysis']['target_type']
        
        # Prepare data once
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.feature_names = self.X.columns.tolist()
        
        # Pre-process numeric/cat or use SupervisedPipeline's logic
        # For simplicity in GA, we use a fixed preprocessing but subset features
        self.population = [Genome(self.problem_type, len(self.feature_names)) for _ in range(pop_size)]
        
    def evolve(self):
        history = []
        
        for gen in range(self.generations):
            print(f"--- Generation {gen+1}/{self.generations} ---")
            
            # 1. Evaluate
            for genome in self.population:
                self._evaluate(genome)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            print(f"Best Fitness: {best.fitness:.4f} (Model: {best.model_choice})")
            
            history.append({
                "gen": gen,
                "best_fitness": best.fitness,
                "best_model": best.model_choice,
                "num_features": sum(best.feature_mask)
            })
            
            # 2. Selection & Reproduction
            new_pop = self.population[:2] # Elitism
            
            while len(new_pop) < self.pop_size:
                # Tournament Selection
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                
                # Crossover
                c1, c2 = self._crossover(p1, p2)
                
                # Mutation
                self._mutate(c1)
                self._mutate(c2)
                
                new_pop.extend([c1, c2])
                
            self.population = new_pop[:self.pop_size]
            
        return history, self.population[0]

    def _evaluate(self, genome: Genome):
        # Subset features
        selected_features = [f for i, f in enumerate(self.feature_names) if genome.feature_mask[i] == 1]
        X_sub = self.X[selected_features]
        
        # Use simple train/test split for speed in GA
        X_train, X_val, y_train, y_val = train_test_split(X_sub, self.y, test_size=0.2, random_state=42)
        
        # Basic imputation/scaling for numeric
        # Handle Categorical if any in subset
        # (Using a very simple pipeline here to avoid overhead)
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        
        if genome.model_choice == "RandomForest":
            model = RandomForestClassifier(**genome.hparams) if self.problem_type == "classification" else RandomForestRegressor(**genome.hparams)
        elif genome.model_choice == "LogisticRegression":
            model = LogisticRegression(**genome.hparams, max_iter=1000)
        else:
            model = LinearRegression()
            
        # Pipeline for simple evaluation
        # Note: In real setup, we'd handle categorical properly. 
        # For now, we only use numeric features in GA feature mask for simplicity if cat is not encoded yet.
        # Let's assume X_sub is already clean or use a minimal imputer.
        # (This is where Layer 2's deterministic logic would be re-used or encapsulated)
        
        # To make it robust, let's use a subset of the SupervisedPipeline's preprocessor logic
        # But for speed in GA, we'll use a simpler version.
        
        # For brevity, let's just use numeric features for the GA Demo
        numeric_sub = X_sub.select_dtypes(include=[np.number])
        if numeric_sub.empty:
            genome.fitness = 0.0
            return
            
        pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
        try:
            pipe.fit(X_train.select_dtypes(include=[np.number]), y_train)
            y_pred = pipe.predict(X_val.select_dtypes(include=[np.number]))
            metrics = calculate_metrics(y_val, y_pred, self.problem_type)
            
            # Multi-objective Fitness: 0.8 * Performance + 0.1 * Simplicty + 0.1 * OverfittingPenalty
            if self.problem_type == "classification":
                perf = metrics.get("accuracy", 0)
            else:
                perf = metrics.get("r2", 0)
                
            complexity_penalty = 0.1 * (1.0 - (sum(genome.feature_mask) / len(self.feature_names)))
            genome.fitness = perf + complexity_penalty
            genome.metrics = metrics
        except:
            genome.fitness = 0.0

    def _tournament_select(self) -> Genome:
        contestants = random.sample(self.population, 3)
        return max(contestants, key=lambda x: x.fitness)

    def _crossover(self, p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
        # Single point crossover on feature mask
        cp = random.randint(0, len(p1.feature_mask)-1)
        c1_mask = p1.feature_mask[:cp] + p2.feature_mask[cp:]
        c2_mask = p2.feature_mask[:cp] + p1.feature_mask[cp:]
        
        c1 = Genome(self.problem_type, len(self.feature_names))
        c2 = Genome(self.problem_type, len(self.feature_names))
        
        c1.feature_mask = c1_mask
        c2.feature_mask = c2_mask
        
        # Inherit model from one parent
        c1.model_choice = p1.model_choice
        c1.hparams = p1.hparams.copy()
        c2.model_choice = p2.model_choice
        c2.hparams = p2.hparams.copy()
        
        return c1, c2

    def _mutate(self, genome: Genome):
        # 1. Feature mutation
        for i in range(len(genome.feature_mask)):
            if random.random() < 0.1:
                genome.feature_mask[i] = 1 - genome.feature_mask[i]
        
        # 2. Hyperparameter mutation
        if genome.model_choice == "RandomForest":
            if random.random() < 0.2:
                genome.hparams["n_estimators"] = max(10, genome.hparams["n_estimators"] + random.randint(-20, 20))
        elif genome.model_choice == "LogisticRegression":
            if random.random() < 0.2:
                genome.hparams["C"] *= random.uniform(0.5, 1.5)
