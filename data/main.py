# main.py

import os
import pandas as pd
from profiling.profiler import DatasetProfiler, TargetIdentifier

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    while True:
        clear_screen()
        print("========================================")
        print("      BEYOND ML LABORATORY v1.0")
        print("========================================")
        print("1. Explore Dataset (Profiler)")
        print("2. Supervised Learning Pipeline")
        print("3. Unsupervised Learning Pipeline")
        print("4. Automatic Experimental Mode")
        print("5. Exit")
        print("========================================")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            explore_dataset()
        elif choice == '2':
            run_supervised()
        elif choice == '3':
            run_unsupervised()
        elif choice == '4':
            run_automatic()
        elif choice == '5':
            print("Exiting BeyondML Laboratory. Goodbye!")
            break
        else:
            input("Invalid choice. Press Enter to continue...")

def load_dataset():
    path = input("Enter the path to your dataset (CSV): ")
    try:
        df = pd.read_csv(path)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset():
    df = load_dataset()
    if df is not None:
        print("\nAnalyzing dataset...")
        # Step 1: Automatic Target Identification (Preliminary)
        identifier = TargetIdentifier(df)
        target_info = identifier.identify()
        
        target_col = target_info.get("suggested_target")
        if target_col:
            print(f"Suggested Target: {target_col} (Confidence: {target_info['confidence_score']:.2f})")
        else:
            print("No clear target column detected.")
            target_col = input("Please specify a target column (or press Enter to skip target-specific analysis): ")
            if target_col == "": target_col = df.columns[0] # Fallback
            
        profiler = DatasetProfiler(df, target_column=target_col)
        profile = profiler.run()
        
        # Display summary (In a real app, this would be a beautiful dashboard)
        print("\n--- Dataset Profile Summary ---")
        print(f"Rows: {profile['metadata']['num_rows']}")
        print(f"Cols: {profile['metadata']['num_columns']}")
        print(f"Numerical Features: {len(profile['feature_types']['numerical'])}")
        print(f"Categorical Features: {len(profile['feature_types']['categorical'])}")
        print(f"Target Type: {profile['target_analysis']['target_type']}")
        
        input("\nPress Enter to return to main menu...")

def run_supervised():
    df = load_dataset()
    if df is None: return
    
    # Step 1: Target Identification
    identifier = TargetIdentifier(df)
    target_info = identifier.identify()
    
    target_col = target_info.get("suggested_target")
    if target_col and target_info['confidence_score'] > 0.6:
        print(f"\nAuto-identified Target: {target_col} (Confidence: {target_info['confidence_score']:.2f})")
    else:
        print("\nCould not confidently auto-identify target.")
        target_col = input(f"Please specify a target column (available: {list(df.columns)}): ")
        if target_col not in df.columns:
            print("Invalid column. Returning to menu.")
            return

    # Step 2: Profiling
    print("\nProfiling dataset for modeling...")
    profiler = DatasetProfiler(df, target_column=target_col)
    profile = profiler.run()
    
    # Step 3: Run Supervised Pipeline
    from modeling.supervised_engine import SupervisedPipeline
    print("\nRunning baseline models...")
    pipeline = SupervisedPipeline(df, target_column=target_col, profile=profile)
    base_results = pipeline.run_baselines()
    
    print("\n--- Baseline Model Results ---")
    for model_name, metrics in base_results.items():
        print(f"Model: {model_name} | Val: {metrics['val_metrics']} | Gap: {metrics['overfitting_gap']:.4f}")

    # Step 4: Evolutionary Optimization (GA)
    print("\n--- [Layer 3] Starting Evolutionary Optimization (GA) ---")
    from modeling.genetic_algorithm import GeneticModelOptimizer
    from evaluation.observability import GAObservability
    
    obs = GAObservability()
    optimizer = GeneticModelOptimizer(df, target_column=target_col, profile=profile, generations=3)
    
    history, best_genome = optimizer.evolve()
    
    for gen_sum in history:
        obs.record_generation(gen_sum)
        
    print("\n--- [GA] Optimization Complete ---")
    print(f"Best Model: {best_genome.model_choice}")
    print(f"Best Fitness: {best_genome.fitness:.4f}")
    print(f"Metrics: {best_genome.metrics}")
    
    report_path = obs.save_report()
    print(f"GA observability report saved to: {report_path}")
    
    input("\nPress Enter to return to main menu...")

def run_unsupervised():
    df = load_dataset()
    if df is None: return
    
    print("\nProfiling dataset for unsupervised learning...")
    profiler = DatasetProfiler(df)
    profile = profiler.run()
    
    from modeling.unsupervised_engine import UnsupervisedPipeline
    print("\nRunning unsupervised analysis (Clustering & PCA)...")
    pipeline = UnsupervisedPipeline(df, profile=profile)
    results = pipeline.run_clustering()
    
    print("\n--- Unsupervised Analysis Results ---")
    for task, metrics in results.items():
        print(f"\nTask: {task}")
        print(f"Metrics: {metrics}")
        
    input("\nPress Enter to return to main menu...")

def run_automatic():
    df = load_dataset()
    if df is None: return
    
    from agentic.orchestrator import AutomaticController
    controller = AutomaticController(df)
    decision = controller.run_auto_pipeline()
    
    if decision['type'] == "supervised":
        target = decision['target']
        # Shared logic from run_supervised could be refactored, but here we inline for simplicity
        print(f"\n[Auto-Branch] Starting Supervised Execution for target: {target}")
        profiler = DatasetProfiler(df, target_column=target)
        profile = profiler.run()
        
        from modeling.supervised_engine import SupervisedPipeline
        pipeline = SupervisedPipeline(df, target_column=target, profile=profile)
        results = pipeline.run_baselines()
        
        print("\n--- Auto-Supervised Results ---")
        for m, met in results.items():
            print(f"{m}: {met['val_metrics']}")
            
    else:
        print("\n[Auto-Branch] Starting Unsupervised Execution")
        profiler = DatasetProfiler(df)
        profile = profiler.run()
        
        from modeling.unsupervised_engine import UnsupervisedPipeline
        pipeline = UnsupervisedPipeline(df, profile=profile)
        results = pipeline.run_clustering()
        
        print("\n--- Auto-Unsupervised Results ---")
        for t, m in results.items():
            print(f"{t}: {m}")

    input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    main_menu()
