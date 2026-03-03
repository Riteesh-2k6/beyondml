"""
Reflection Agent — Evaluates output for overfitting/dominance and iterates pipeline.
"""

import json
from typing import Dict, Any, Callable, Awaitable
from ..llm.base import LLMProvider

SYSTEM_PROMPT = """You are an AI ML Supervisor Reflection Agent.
Your job is to review model performance and propose concrete pandas dataframe expressions to fix issues like overfitting or feature dominance.

Respond with a JSON object:
{
  "reasoning": "Explain why the model is struggling (e.g., 'Model is overfitting due to high variance' or 'Feature X is dominating').",
  "features_to_drop": ["column_to_remove1"],
  "new_features": [
    {
      "name": "new_col_name",
      "expr": "pandas expression (e.g., df['A'] * df['B'])",
      "rationale": "why this helps"
    }
  ],
  "next_model": "RandomForest|LogisticRegression|GradientBoosting|SVM|DecisionTree|KNN",
  "next_ga_generations": 5,
  "next_ga_pop_size": 10
}

Rules for features_to_drop:
- YOU HAVE EXPLICIT PERMISSION to ruthlessly prune ANY feature you consider 'too good to be true', highly biased, or excessively dominating model predictions (e.g., target feature leaks).
- Drop features that prevent the model from generalizing well.

Rules for new_features:
- Use only valid vectorized pandas/numpy expressions.
- Reference existing columns exactly as named.
- Use df['column_name'] syntax.
"""

class ReflectionAgent:
    """Agent that decides whether to loop pipeline based on metrics and proposes changes."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        eval_result: Dict[str, Any],
        iteration: int,
        max_iterations: int,
        log: Callable[[str], Awaitable[None]]
    ) -> Dict[str, Any]:
        await log(f"\n[bold magenta]● Reflection Agent[/bold magenta]  Evaluating Iteration {iteration} / {max_iterations}...")

        test_score = eval_result.get("test_score", 0)
        train_score = eval_result.get("train_score", 0)
        importances = eval_result.get("feature_importances", {})

        # Rules requested by user
        overfit_gap = train_score - test_score
        is_overfitting = overfit_gap > 0.10
        
        max_imp = max(importances.values()) if importances else 0
        dominating_feat = next((k for k, v in importances.items() if v == max_imp), None)
        is_dominating = max_imp > 0.50

        is_underperforming = test_score < 0.85

        # Detect clear data leakage (Perfect score usually means target column leak)
        is_leaking = test_score >= 0.999

        needs_improvement = is_overfitting or is_dominating or is_underperforming or is_leaking

        if not needs_improvement:
             await log("  [bold green]✓ Model performance is satisfactory![/bold green] No further iterations needed.")
             return {"status": "satisfied", "modifications": None}

        if iteration >= max_iterations:
             await log(f"  [yellow]⚠ Max iterations ({max_iterations}) reached.[/yellow] Stopping reflection loop.")
             return {"status": "satisfied", "modifications": None}

        # Build prompt for LLM explaining the failure points
        issues = []
        if is_overfitting:
            issues.append(f"OVERFITTING DETECTED (Train {train_score:.3f} vs Test {test_score:.3f}).")
            await log(f"  [red]Overfitting detected:[/red] train-test gap is {overfit_gap:.3f}")
        if is_dominating:
            issues.append(f"FEATURE DOMINANCE DETECTED ('{dominating_feat}' contributes {max_imp:.3f}!).")
            await log(f"  [red]Feature Dominance:[/red] '{dominating_feat}' has {max_imp:.3f} importance.")
        if is_underperforming and not is_overfitting:
            issues.append(f"UNDERPERFORMING (Test score {test_score:.3f} < 0.85 Target).")
            await log(f"  [yellow]Goal Unmet:[/yellow] Test score {test_score:.3f} is below 0.85")
        if is_leaking:
            issues.append(f"DATA LEAKAGE DETECTED (Test score {test_score:.3f} >= 0.999 is suspiciously perfect. A feature is likely leaking the target variable).")
            await log(f"  [red]Data Leak Suspected:[/red] Test score is suspiciously perfect ({test_score:.3f}).")

        msg = (
            f"Current Test Score: {test_score:.3f} | Train Score: {train_score:.3f}\n"
            f"Top Features: {json.dumps(dict(list(sorted(importances.items(), key=lambda x: x[1], reverse=True))[:5]))}\n"
            f"Issues to fix: {' '.join(issues)}\n"
            f"Determine the absolute best optimal solution path. Suggest 1-3 new interaction features or log transformations, list any features to drop, and pick a better algorithm (next_model) or modify GA trainer iterations if appropriate."
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                json_mode=True,
            )
            result = json.loads(response)
            
            reasoning = result.get("reasoning", "")
            await log(f"  [bold cyan]AI Reflection:[/bold cyan]")
            await log(f"    [italic]{reasoning}[/italic]")
            
            drops = result.get("features_to_drop", [])
            for d in drops:
                 await log(f"  [red]−[/red] Proposed Drop: {d}")
                 
            new_feats = result.get("new_features", [])
            for f in new_feats:
                 await log(f"  [green]+[/green] Proposed Feature: [bold]{f.get('name')}[/bold]")
                 await log(f"    [dim]Formula: {f.get('expr')}[/dim]")
                 
            next_model = result.get("next_model")
            next_ga_gens = result.get("next_ga_generations", 5)
            next_ga_pop = result.get("next_ga_pop_size", 10)
            
            if next_model:
                 await log(f"  [cyan]⟳[/cyan] Proposed Algorithm Switch: {next_model}")
                 await log(f"  [cyan]⟳[/cyan] Proposed GA Params: Pop Size {next_ga_pop} | Generations {next_ga_gens}")

            return {
                "status": "needs_improvement",
                "modifications": {
                    "features_to_drop": drops,
                    "new_features": new_feats,
                    "next_model": next_model,
                    "next_ga_generations": next_ga_gens,
                    "next_ga_pop_size": next_ga_pop
                }
            }
        except Exception as e:
            await log(f"  [dim]⚠ Reflection error: {e}[/dim]")
            return {"status": "satisfied", "modifications": None}
