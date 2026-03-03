# orchestrator.py

from typing import Dict, Any
import pandas as pd
from profiling.profiler import DatasetProfiler, TargetIdentifier
from agentic.reasoning_engine import SimulatedReasoningEngine

class AutomaticController:
    """
    Reconciles deterministic confidence scores with LLM recommendations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_auto_pipeline(self):
        print("\n--- [Layer 5] Initiating Automatic Experimental Mode ---")
        
        # 1. Deterministic Data Analysis
        print("Running deterministic target identification...")
        identifier = TargetIdentifier(self.df)
        target_info = identifier.identify()
        
        print("Generating dataset profile...")
        profiler = DatasetProfiler(self.df)
        profile = profiler.run()
        
        # 2. Agentic Reasoning Layer
        print("Activating Strategic Reasoning Engine (LLM)...")
        reasoner = SimulatedReasoningEngine(profile, target_info)
        decision = reasoner.analyze_intent()
        
        print("\n=== Agentic Decision Report ===")
        print(f"Detected Problem Type: {decision['problem_type'].upper()}")
        print(f"Reasoning: {decision['reasoning_justification']}")
        print(f"Confidence: {decision['confidence_level']}")
        print(f"Insights: {decision['semantic_insights']}")
        
        # 3. Execution Path Selection
        if decision['problem_type'] == "supervised":
            target = decision['recommended_target']
            print(f"\nProceeding to Supervised Pipeline with target: {target}")
            # In a real app, we'd call the supervised pipeline logic here
            return {"type": "supervised", "target": target}
        else:
            print("\nProceeding to Unsupervised Pipeline...")
            return {"type": "unsupervised"}
