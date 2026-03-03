"""
Feature Engineering Agent — LLM proposes derived features, user approves.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Awaitable, List
from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are an ML feature engineer. Given column names, dtypes, and EDA insights,
propose 2-4 derived features that could improve model signal.

You MUST respond with a JSON object:
{
  "features": [
    {
      "name": "new_column_name",
      "expr": "pandas expression using df (e.g. df['col1'] * df['col2'])",
      "rationale": "why this feature adds signal"
    }
  ]
}

Rules:
- Only use vectorised pandas/numpy expressions
- Only reference columns that exist in the dataset
- No imports, no builtins, no file I/O
- Keep expressions simple: arithmetic, ratios, log transforms
- Use df['column_name'] syntax
"""


class FeatureAgent:
    """LLM-powered feature engineering with sandboxed eval."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        eda_insights: List[Dict],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log("[bold orange1]● Feature Agent[/bold orange1]  Proposing derived features...")

        proposals = await self._ask_llm(df, profile, eda_insights, log)
        features_applied = []
        features_rejected = []

        for feat in proposals:
            name = feat.get("name", "unnamed")
            expr = feat.get("expr", "")
            rationale = feat.get("rationale", "No rationale provided")

            try:
                # Sandboxed eval — only df, np, pd in scope
                result = eval(expr, {"__builtins__": {}}, {"df": df, "np": np, "pd": pd})
                if isinstance(result, pd.Series) and len(result) == len(df):
                    df[name] = result
                    features_applied.append(name)
                    await log(f"  [green]+[/green] {name} = {expr}")
                    await log(f"    [bold cyan]Explanation:[/bold cyan] [dim]{rationale}[/dim]")
                else:
                    features_rejected.append(name)
                    await log(f"  [red]✗[/red] {name} — invalid result shape")
            except Exception as e:
                features_rejected.append(name)
                await log(f"  [red]✗[/red] {name} — eval failed: {e}")

        if features_applied:
            await log(f"\n  [green]✓ {len(features_applied)} features applied. Shape: {df.shape}[/green]")
        else:
            await log("  [yellow]No features applied.[/yellow]")

        return {
            "df": df,
            "features_applied": features_applied,
            "features_rejected": features_rejected,
            "feature_proposals": proposals,
        }

    async def _ask_llm(self, df, profile, eda_insights, log) -> List[Dict]:
        try:
            cols_info = {col: str(df[col].dtype) for col in df.columns}
            msg = (
                f"Columns: {json.dumps(cols_info)}\n"
                f"Shape: {df.shape}\n"
                f"Numerical stats: {json.dumps(profile.get('numerical_summary', {}), indent=2)[:1500]}\n"
                f"EDA insights: {json.dumps(eda_insights[:5], default=str)[:1000]}"
            )
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                json_mode=True,
            )
            result = json.loads(response)
            return result.get("features", [])
        except Exception as e:
            await log(f"  [dim]⚠ LLM error: {e}[/dim]")
            return []
