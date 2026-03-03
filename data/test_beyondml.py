# test_beyondml.py

import pandas as pd
from profiling.profiler import DatasetProfiler, TargetIdentifier
from modeling.supervised_engine import SupervisedPipeline
from modeling.unsupervised_engine import UnsupervisedPipeline
from modeling.genetic_algorithm import GeneticModelOptimizer
from agentic.orchestrator import AutomaticController

def test_pipeline():
    print("Loading test data...")
    df = pd.read_csv('echallan_daily_data.csv')
    
    # 1. Target Identification
    print("\n--- Testing Target Identification ---")
    identifier = TargetIdentifier(df)
    target_info = identifier.identify()
    print(f"Suggested Target: {target_info['suggested_target']} (Confidence: {target_info['confidence_score']:.2f})")
    target = target_info['suggested_target']
    
    # 2. Profiling
    print("\n--- Testing Profiler ---")
    profiler = DatasetProfiler(df, target_column=target)
    profile = profiler.run()
    print("Metadata:", profile['metadata'])
    print("Feature Types:", profile['feature_types'])
    
    # 3. Supervised Baselines
    print("\n--- Testing Supervised Baselines ---")
    supervised = SupervisedPipeline(df, target_column=target, profile=profile)
    base_results = supervised.run_baselines()
    for m, res in base_results.items():
        print(f"Model: {m}, Val Accuracy: {res['val_metrics'].get('accuracy', 'N/A')}")
        
    # 4. GA Optimization
    print("\n--- Testing GA (1 generation) ---")
    optimizer = GeneticModelOptimizer(df, target_column=target, profile=profile, generations=1, pop_size=2)
    history, best_genome = optimizer.evolve()
    print(f"GA Best Fitness: {best_genome.fitness:.4f}")
    
    # 5. Unsupervised
    print("\n--- Testing Unsupervised ---")
    unsupervised = UnsupervisedPipeline(df, profile=profile)
    un_results = unsupervised.run_clustering()
    print(f"KMeans (k=2) Silhouette: {un_results['KMeans (k=2)']['silhouette_score']:.4f}")
    
    # 6. Automatic Mode
    print("\n--- Testing Automatic Mode ---")
    controller = AutomaticController(df)
    decision = controller.run_auto_pipeline()
    print(f"Auto-Decision: {decision['type']}")

if __name__ == "__main__":
    test_pipeline()
