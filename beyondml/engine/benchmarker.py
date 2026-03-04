"""
PMLB Benchmarking Suite for BeyondML.
Runs the autonomous pipeline on standardized datasets to measure effectiveness.
"""

import pandas as pd
import numpy as np
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pmlb import fetch_data, dataset_names

# Load API keys
load_dotenv()
from beyondml.engine.profiler import DatasetProfiler
from beyondml.engine.genetic import GeneticModelOptimizer
from beyondml.agents.orchestrator import OrchestratorAgent
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.llm import get_llm_provider

class PMLBRunner:
    """Benchmark runner for BeyondML using PMLB datasets."""

    def __init__(self, datasets: List[str] = None):
        if datasets is None:
            self.datasets = ["titanic", "car"] # Lite version for demo
        else:
            self.datasets = datasets
        self.results = []

    async def run_benchmark(self):
        print("Starting PMLB Benchmark Suite...")
        llm = get_llm_provider()
        
        for name in self.datasets:
            print(f"\n[Benchmark: {name}] Fetching data...")
            try:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir='./data/pmlb_cache')
                df = pd.DataFrame(X)
                df['target'] = y
                
                print(f"  Shape: {df.shape}")
                
                # 1. Profile
                profiler = DatasetProfiler(df, target_column='target')
                profile = profiler.run()
                ori = profile.get('overfitting_risk_index', {}).get('score', 0.5)
                print(f"  ORI Score: {ori:.4f}")

                # 2. GA Optimization (Run a compact search for benchmarks)
                optimizer = GeneticModelOptimizer(
                    df=df,
                    target_column='target',
                    profile=profile,
                    pop_size=5,
                    generations=2,
                    model_choice="RandomForest"
                )
                
                print(f"  Running GA Evolution...")
                history, best = optimizer.evolve()
                
                res = {
                    "dataset": name,
                    "ori": ori,
                    "best_fitness": best.fitness,
                    "mu_cv": best.metrics.get("mu_cv", 0),
                    "sigma_cv": best.metrics.get("sigma_cv", 0),
                    "gap": best.metrics.get("gap", 0),
                    "model": best.model_choice
                }
                self.results.append(res)
                print(f"  Best Mu_CV: {res['mu_cv']:.4f} | Fitness: {best.fitness:.4f}")
                
            except Exception as e:
                print(f"  Error on {name}: {e}")

        self._export_results()

    def _export_results(self):
        df_res = pd.DataFrame(self.results)
        print("\nBENCHMARK SUMMARY")
        print(df_res.to_string())
        
        # Save to markdown artifact-like format for the user
        report_path = "benchmark_results.md"
        with open(report_path, "w") as f:
            f.write("# BeyondML Benchmark Report (PMLB)\n\n")
            f.write("| Dataset | ORI | Best Mu_CV | Fitness | Gap | Model |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in self.results:
                f.write(f"| {r['dataset']} | {r['ori']:.3f} | {r['mu_cv']:.4f} | {r['best_fitness']:.4f} | {r['gap']:.4f} | {r['model']} |\n")
        
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    runner = PMLBRunner()
    asyncio.run(runner.run_benchmark())
