"""
Evaluator Agent — LLM-powered result interpretation and model evaluation.
"""

import json
import joblib
import os
from typing import Dict, Any, Callable, Awaitable
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from ..llm.base import LLMProvider
from ..engine.supervised import SupervisedPipeline


SYSTEM_PROMPT = """You are an ML evaluation expert. Given model training results,
provide a brief, insightful interpretation.

Respond with a JSON object:
{
  "narration": "3-5 sentence interpretation of the results, mentioning strengths and weaknesses",
  "recommendations": ["actionable next step 1", "actionable next step 2"],
  "overall_assessment": "excellent|good|fair|poor"
}

Be specific: reference actual metric values and feature names.
"""


class EvaluatorAgent:
    """Final model evaluation with LLM narration."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        profile: Dict[str, Any],
        best_params: Dict[str, Any],
        model_type: str,
        problem_type: str,
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log("[bold blue]● Evaluator[/bold blue]  Training final model and evaluating...")

        # Build the final model with best params
        model_obj = self._build_model(model_type, best_params, problem_type)

        # Train and evaluate
        pipeline = SupervisedPipeline(df, target_column, profile)
        trained_pipe, train_metrics, test_metrics, importances = pipeline.train_final_model(model_obj)

        # Log metrics
        await log(f"\n  [bold green]Metrics (Train / Test):[/bold green]")
        for k in test_metrics:
            train_v = train_metrics.get(k, 0)
            test_v = test_metrics.get(k, 0)
            if isinstance(train_v, float):
                await log(f"    {k}: Train [dim]{train_v:.4f}[/dim] | Test [bold]{test_v:.4f}[/bold]")
            else:
                await log(f"    {k}: Train {train_v} | Test {test_v}")

        # Feature importances
        if importances:
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            await log(f"\n  [bold]Top Feature Importances:[/bold]")
            for name, imp in sorted_imp:
                bar = "█" * int(imp * 40)
                await log(f"    {name[:20]:20s} {bar} {imp:.3f}")

        # Save model
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_best.pkl")
        joblib.dump(trained_pipe, model_path)
        await log(f"\n  Model saved: [bold]{model_path}[/bold]")

        # LLM narration
        narration = await self._narrate(test_metrics, importances, model_type, problem_type, log)

        # Get primary test and train scores
        if problem_type == "classification":
            test_score = test_metrics.get("accuracy", 0)
            train_score = train_metrics.get("accuracy", 0)
        else:
            test_score = test_metrics.get("r2", 0)
            train_score = train_metrics.get("r2", 0)

        return {
            "test_score": test_score,
            "train_score": train_score,
            "eval_report": test_metrics,
            "feature_importances": importances,
            "eval_narration": narration,
            "model_path": model_path,
            "best_params": best_params,
        }

    def _build_model(self, model_type: str, params: Dict, problem_type: str):
        clean_params = {k: v for k, v in params.items() if k not in ("max_iter",)}

        if model_type == "RandomForest":
            if problem_type == "classification":
                return RandomForestClassifier(**clean_params, random_state=42)
            return RandomForestRegressor(**clean_params, random_state=42)
        elif model_type == "LogisticRegression":
            return LogisticRegression(**params, max_iter=1000)
        elif model_type == "LinearRegression":
            return LinearRegression()
        else:
            # Fallback
            if problem_type == "classification":
                return RandomForestClassifier(random_state=42)
            return RandomForestRegressor(random_state=42)

    async def _narrate(self, metrics, importances, model_type, problem_type, log) -> str:
        try:
            msg = (
                f"Model: {model_type}\n"
                f"Problem: {problem_type}\n"
                f"Metrics: {json.dumps(metrics)}\n"
                f"Top features: {json.dumps(dict(list(sorted(importances.items(), key=lambda x: x[1], reverse=True))[:5]))}"
            )
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                json_mode=True,
            )
            result = json.loads(response)
            narration = result.get("narration", "")
            if narration:
                await log(f"\n  [italic]{narration}[/italic]")
            recs = result.get("recommendations", [])
            if recs:
                await log(f"\n  [bold]Recommendations:[/bold]")
                for r in recs:
                    await log(f"    • {r}")
            return narration
        except Exception as e:
            await log(f"  [dim]⚠ Narration error: {e}[/dim]")
            return ""
