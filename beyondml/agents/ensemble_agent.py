"""
Ensemble Agent — Combines top-N GA genomes into a stacking or voting ensemble.
"""

from typing import Dict, Any, Callable, Awaitable, List
import pandas as pd
from ..llm.base import LLMProvider
from ..engine.ensemble import EnsembleEngine
from ..engine.supervised import SupervisedPipeline


class EnsembleAgent:
    """Trains ensemble models from top-N GA-evolved genomes."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        profile: Dict[str, Any],
        top_genomes: list,
        problem_type: str,
        log: Callable[[str], Awaitable[None]],
        strategy: str = "stacking",
    ) -> Dict[str, Any]:
        await log(f"[bold yellow]● Ensemble Agent[/bold yellow]  Building {strategy} ensemble from top-{len(top_genomes)} genomes...")

        engine = EnsembleEngine(problem_type)

        # Log base model composition
        model_names = [g.model_choice for g in top_genomes]
        await log(f"  Base models: {', '.join(model_names)}")

        # Build ensemble
        if strategy == "stacking":
            ensemble_model = engine.build_stacking(top_genomes)
            await log("  Strategy: Stacking (meta-learner: LogisticRegression)")
        else:
            ensemble_model = engine.build_voting(top_genomes)
            await log("  Strategy: Soft Voting")

        # Train and evaluate via SupervisedPipeline
        pipeline = SupervisedPipeline(df, target_column, profile)
        trained_pipe, train_metrics, test_metrics, importances = pipeline.train_final_model(ensemble_model)

        # Log results
        await log(f"\n  [bold green]Ensemble Results:[/bold green]")
        for k in test_metrics:
            train_v = train_metrics.get(k, 0)
            test_v = test_metrics.get(k, 0)
            if isinstance(train_v, float):
                await log(f"    {k}: Train [dim]{train_v:.4f}[/dim] | Test [bold]{test_v:.4f}[/bold]")

        # Primary score
        if problem_type == "classification":
            test_score = test_metrics.get("accuracy", 0)
            train_score = train_metrics.get("accuracy", 0)
        else:
            test_score = test_metrics.get("r2", 0)
            train_score = train_metrics.get("r2", 0)

        await log(f"\n  [bold green]✓ Ensemble Complete![/bold green] Test: {test_score:.4f}")

        return {
            "test_score": test_score,
            "train_score": train_score,
            "eval_report": test_metrics,
            "strategy": strategy,
            "base_models": model_names,
            "model_type": f"Ensemble({strategy})",
        }
