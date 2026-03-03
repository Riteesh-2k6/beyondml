"""
Outlier Handler Agent — detects outliers and lets user choose strategy.

LLM recommends a strategy, user can override. Human-in-the-loop.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Awaitable, Optional, List
from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are a data cleaning expert. Given outlier information from a dataset,
recommend the best outlier handling strategy.

You MUST respond with a JSON object:
{
  "recommended_strategy": "remove|cap|log|flag|keep",
  "reasoning": "1-2 sentence explanation",
  "per_column_notes": {"col": "specific note about this column's outliers"}
}

Strategies:
- remove: drop outlier rows entirely
- cap: clip at IQR bounds (Q1-1.5*IQR, Q3+1.5*IQR)
- log: np.log1p transform on flagged columns
- flag: add binary is_outlier column (preserves data)
- keep: leave unchanged (if outliers are informative)
"""


class OutlierAgent:
    """Outlier detection and handling with LLM recommendation + user override."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df: pd.DataFrame,
        outlier_summary: Dict[str, Any],
        profile: Dict[str, Any],
        log: Callable[[str], Awaitable[None]],
        get_user_input: Callable[[str], Awaitable[str]],
    ) -> Dict[str, Any]:
        """
        Args:
            get_user_input: Async function that prompts user and returns their response.
                           This is the human-in-the-loop mechanism.
        """
        # Find columns with outliers
        outlier_cols = [col for col, info in outlier_summary.items() if info.get("count", 0) > 0]

        if not outlier_cols:
            await log("  [green]✓ No significant outliers detected.[/green]")
            return {"outlier_strategy": "keep", "outlier_columns": [], "df": df}

        total_outliers = sum(outlier_summary[c]["count"] for c in outlier_cols)
        await log(f"[bold yellow]● Outlier Handler[/bold yellow]  {total_outliers} outlier rows in: {outlier_cols}")

        # Ask LLM for recommendation
        recommendation = await self._ask_llm(outlier_summary, outlier_cols, log)
        rec_strategy = recommendation.get("recommended_strategy", "cap")
        reasoning = recommendation.get("reasoning", "")

        await log(f"  Groq recommends: [bold cyan]{rec_strategy}[/bold cyan] for {outlier_cols}")
        if reasoning:
            await log(f"  [bold cyan]Explanation:[/bold cyan]")
            await log(f"    [dim]{reasoning}[/dim]")

        # Present options to user (human-in-the-loop)
        prompt = (
            f"\nOutlier strategy (Groq recommends: {rec_strategy}):\n"
            f"    [1] remove   - drop outlier rows\n"
            f"    [2] cap      - clip at IQR bounds\n"
            f"    [3] log      - np.log1p transform\n"
            f"    [4] flag     - add binary is_outlier column\n"
            f"    [5] keep     - leave unchanged\n"
            f"Enter 1-5 (or press Enter for recommendation):"
        )
        await log(prompt)
        user_input = await get_user_input("Your response...")

        strategy_map = {"1": "remove", "2": "cap", "3": "log", "4": "flag", "5": "keep"}
        strategy = strategy_map.get(user_input.strip(), rec_strategy)

        await log(f"\n  Applying strategy: [bold green]{strategy}[/bold green]")

        # Apply strategy
        df_result = self._apply_strategy(df.copy(), strategy, outlier_summary, outlier_cols)

        await log(f"  [green]✓ Outlier handling complete. Shape: {df_result.shape}[/green]")

        return {
            "outlier_strategy": strategy,
            "outlier_columns": outlier_cols,
            "df": df_result,
        }

    def _apply_strategy(
        self, df: pd.DataFrame, strategy: str, outlier_summary: Dict, outlier_cols: List[str]
    ) -> pd.DataFrame:
        if strategy == "remove":
            all_indices = set()
            for col in outlier_cols:
                all_indices.update(outlier_summary[col].get("indices", []))
            df = df.drop(index=list(all_indices), errors="ignore")

        elif strategy == "cap":
            for col in outlier_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    df[col] = df[col].clip(lower, upper)

        elif strategy == "log":
            for col in outlier_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])

        elif strategy == "flag":
            for col in outlier_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[f"is_outlier_{col}"] = ((df[col] - mean).abs() / std > 2.5).astype(int)

        # "keep" = do nothing
        return df

    async def _ask_llm(self, outlier_summary: Dict, outlier_cols: List[str], log) -> Dict[str, Any]:
        try:
            msg = f"Outlier columns: {outlier_cols}\nDetails: {json.dumps(outlier_summary, default=str)[:2000]}"
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                json_mode=True,
            )
            return json.loads(response)
        except Exception as e:
            await log(f"  [dim]⚠ LLM error: {e}[/dim]")
            return {"recommended_strategy": "cap", "reasoning": "Default: cap outliers at IQR bounds"}
