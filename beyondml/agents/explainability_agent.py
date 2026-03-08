"""
Explainability Agent — provides human-readable SHAP-based model interpretations.
"""

import json
from typing import Dict, Any, Callable, Awaitable
import pandas as pd
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are an ML explainability expert. You will receive a summary of SHAP feature importance scores for a trained model.

Your goal is to provide a brief, insightful narrative explaining what these features mean in the context of the dataset and how they influence the model's predictions.
Focus on the top 3-5 features. Write in a clear, business-friendly tone.

Respond with a JSON object:
{
  "explanation": "3-4 sentences explaining the most important features driving the model.",
  "key_drivers": ["Driver 1: explanation", "Driver 2: explanation"]
}
"""

class ExplainabilityAgent:
    """Agent that generates SHAP values and an LLM narrative for global model explainability."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.name = "Explainability Agent"

    async def run(
        self,
        model_pipeline: Any,
        X_eval: pd.DataFrame,
        target_column: str,
        problem_type: str,
        log: Callable[[str], Awaitable[None]]
    ) -> Dict[str, Any]:
        """Extract SHAP values and generate narrative."""
        await log(f"\n[bold magenta]● {self.name}[/bold magenta] Analyzing model decisions...")

        if not SHAP_AVAILABLE:
            await log("  [yellow]⚠ SHAP library not installed. Skipping explainability.[/yellow]")
            return {"status": "skipped", "reason": "SHAP not installed"}

        try:
            # We need to extract the preprocessor and the actual model
            # Assuming model_pipeline is a scikit-learn Pipeline with ('preprocessor', 'model')
            if hasattr(model_pipeline, "named_steps") and "model" in model_pipeline.named_steps:
                preprocessor = model_pipeline.named_steps.get("preprocessor")
                model = model_pipeline.named_steps["model"]
                
                # Transform data
                if preprocessor:
                    X_transformed = preprocessor.transform(X_eval)
                    
                    # Try to get feature names
                    if hasattr(preprocessor, "get_feature_names_out"):
                        feature_names = preprocessor.get_feature_names_out()
                        # Clean feature names
                        feature_names = [f.split("__")[-1] for f in feature_names]
                    else:
                        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
                else:
                    X_transformed = X_eval
                    feature_names = X_eval.columns.tolist()
            else:
                # Not a standard pipeline
                model = model_pipeline
                X_transformed = X_eval
                feature_names = X_eval.columns.tolist()

            # For speed and stability, sample the data for SHAP
            sample_size = min(100, X_transformed.shape[0])
            
            # Subsample data
            if isinstance(X_transformed, pd.DataFrame):
                X_sample = X_transformed.sample(sample_size, random_state=42)
            else:
                idx = np.random.choice(X_transformed.shape[0], sample_size, replace=False)
                X_sample = X_transformed[idx]

            # Use appropriate explainer
            await log("  [dim]Calculating SHAP values...[/dim]")
            try:
                # TreeExplainer is fast but only works for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                # Fallback for linear/neural models (KernelExplainer or LinearExplainer)
                # Using LinearExplainer for logistic/linear models
                try:
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    # Final fallback: KernelExplainer (slow, so we use tiny sample)
                    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_sample, 10))
                    shap_values = explainer.shap_values(X_sample[:50])

            # Handle multi-class output format for shap_values
            if isinstance(shap_values, list):
                # For classification, we usually look at the positive class (index 1)
                if len(shap_values) > 1:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # Shape is (samples, features, classes), take index 1 if available
                if shap_values.shape[2] > 1:
                    shap_values = shap_values[:, :, 1]
                else:
                    shap_values = shap_values[:, :, 0]

            # Calculate global feature importance (mean absolute SHAP)
            global_shap = np.abs(shap_values).mean(axis=0)
            
            # Map back to feature names
            importance_dict = {name: float(val) for name, val in zip(feature_names, global_shap)}
            
            # Sort by importance
            sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_imp[:10])

            # Print ASCII Bar Chart
            await log(f"\n  [bold green]SHAP Feature Importance (Top Features):[/bold green]")
            max_val = max(top_features.values()) if top_features else 1.0
            for name, val in top_features.items():
                if max_val > 0:
                    bar_len = int((val / max_val) * 30)
                    bar = "█" * max(1, bar_len)
                else:
                    bar = ""
                await log(f"    {name[:25]:25s} {bar} ({val:.4f})")

            # Generate narrative
            await log("\n  [dim]Generating narrative explanation...[/dim]")
            msg = (
                f"Problem Type: {problem_type}\n"
                f"Top features and their SHAP values (mean absolute impact on prediction):\n"
                f"{json.dumps(top_features, indent=2)}"
            )
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                json_mode=True,
            )
            result = json.loads(response)
            
            explanation = result.get("explanation", "")
            if explanation:
                await log(f"\n  [italic]{explanation}[/italic]")
                
            drivers = result.get("key_drivers", [])
            for d in drivers:
                await log(f"  • {d}")

            return {
                "status": "success",
                "top_features": top_features,
                "explanation": explanation,
                "drivers": drivers
            }

        except Exception as e:
            await log(f"  [red]⚠ Could not generate SHAP explanations: {e}[/red]")
            import traceback
            tb_str = traceback.format_exc()
            return {"status": "error", "reason": str(e), "traceback": tb_str}
