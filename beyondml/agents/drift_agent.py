"""
Data Drift Agent — Detects distributional shifts between reference and current datasets.
"""

import json
from typing import Dict, Any, Callable, Awaitable
import pandas as pd
import numpy as np
from scipy import stats

from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are an ML Data Quality expert. You are provided with a data drift report comparing a reference (training) dataset to a current (inference) dataset.
You will see a list of features that have drifted, their data types, and their p-values from statistical tests (KS test for numerical, Chi-Square for categorical).

Your goal is to provide a brief, insightful narrative on the implications of this drift for model reliability.
If no features drifted, state that the data distribution is stable.
If features drifted, explain potential risks (e.g., model degradation, concept drift) and suggest what the user should do (e.g., retrain, investigate data pipeline).
Write in a clear, business-friendly tone.

Respond with a JSON object:
{
  "drift_narrative": "3-4 sentences explaining the severity and implications of the drift.",
  "recommended_actions": ["Action 1", "Action 2"]
}
"""

class DriftAgent:
    """Agent that detects data drift between reference and current datasets."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.name = "Data Drift Agent"

    async def run(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        log: Callable[[str], Awaitable[None]],
        p_value_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect drift and generate narrative."""
        await log(f"\n[bold magenta]● {self.name}[/bold magenta] Analyzing data distribution for drift...")

        drift_results = {}
        common_columns = list(set(df_reference.columns) & set(df_current.columns))

        if not common_columns:
            await log("  [red]⚠ No common columns found between reference and current datasets.[/red]")
            return {"status": "error", "reason": "No common columns"}

        await log(f"  [dim]Comparing {len(common_columns)} features...[/dim]")

        # Perform statistical tests
        for col in sorted(common_columns):
            ref_col = df_reference[col].dropna()
            cur_col = df_current[col].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                continue

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
                # Use Kolmogorov-Smirnov test for continuous data
                try:
                    statistic, p_value = stats.ks_2samp(ref_col, cur_col)
                    is_drifted = p_value < p_value_threshold
                    drift_results[col] = {
                        "type": "numeric",
                        "test": "KS",
                        "p_value": float(p_value),
                        "drifted": bool(is_drifted)
                    }
                except Exception:
                    pass
            else:
                # Use Chi-Square test for categorical data
                try:
                    # Get frequencies
                    val_counts_ref = ref_col.value_counts().to_dict()
                    val_counts_cur = cur_col.value_counts().to_dict()
                    
                    # Align keys
                    all_keys = list(set(val_counts_ref.keys()) | set(val_counts_cur.keys()))
                    
                    # Ensure minimum count of 1 for Chi-Square stability
                    ref_freqs = [val_counts_ref.get(k, 0) + 1 for k in all_keys]
                    cur_freqs = [val_counts_cur.get(k, 0) + 1 for k in all_keys]
                    
                    statistic, p_value = stats.chisquare(f_obs=cur_freqs, f_exp=ref_freqs)
                    is_drifted = p_value < p_value_threshold
                    drift_results[col] = {
                        "type": "categorical",
                        "test": "Chi-Square",
                        "p_value": float(p_value),
                        "drifted": bool(is_drifted)
                    }
                except Exception:
                    pass

        # Filter for drifted features
        drifting_features = {k: v for k, v in drift_results.items() if v["drifted"]}
        
        # Logging results
        if not drifting_features:
            await log("  [bold green]✓ No significant data drift detected.[/bold green]")
        else:
            await log(f"  [bold yellow]⚠ Detected drift in {len(drifting_features)} features:[/bold yellow]")
            for col, info in drifting_features.items():
                p_val_str = f"{info['p_value']:.4e}" if info['p_value'] < 0.0001 else f"{info['p_value']:.4f}"
                await log(f"    • {col} ({info['type']}) - p={p_val_str} ({info['test']})")

        # Generate narrative
        await log("\n  [dim]Generating drift implications narrative...[/dim]")
        
        msg = (
            f"Total features compared: {len(common_columns)}\n"
            f"Features drifted: {len(drifting_features)}\n"
            f"Details of drifted features:\n"
            f"{json.dumps(drifting_features, indent=2)}\n"
            f"Significance threshold (p-value): {p_value_threshold}"
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
            
            narrative = result.get("drift_narrative", "")
            if narrative:
                await log(f"\n  [italic]{narrative}[/italic]")
                
            actions = result.get("recommended_actions", [])
            for a in actions:
                await log(f"  • {a}")

            return {
                "status": "success",
                "drifting_features": drifting_features,
                "all_results": drift_results,
                "drift_narrative": narrative,
                "recommended_actions": actions
            }

        except Exception as e:
            await log(f"  [red]⚠ Could not generate Drift narrative: {e}[/red]")
            import traceback
            tb_str = traceback.format_exc()
            return {"status": "error", "reason": str(e), "traceback": tb_str}
