import json
from typing import Dict, Any, List, Callable, Awaitable
from ..llm.base import LLMProvider

SYSTEM_PROMPT = """You are a Data Imputation Expert. Your role is to recommend the best strategy for filling in missing values (NaNs) or identified logical impossibilities.

Given the column metadata and user description, choose the most scientifically robust imputation strategy.

Strategies:
- `mean`: Use the average (best for normal distributions).
- `median`: Use the middle value (best for skewed data/outliers).
- `mode`: Use the most frequent value (best for categorical data).
- `constant`: Use a specific logical value (e.g., 0 or "Unknown").
- `drop`: If the column is too corrupted (more than 70% missing).

You MUST respond with a JSON object:
{
  "strategies": [
    {
      "column": "column_name",
      "strategy": "mean" or "median" or "mode" or "constant" or "drop",
      "fill_value": null or specific_value,
      "reasoning": "Scientific justification"
    }
  ]
}
"""

class ImputationAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df_summary: str,
        missing_report: Dict[str, Any],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log("[bold blue]● Imputation Agent[/bold blue]  Recommending cleaning strategies...")
        
        user_msg = f"""Dataset Context:
{df_summary}

Missing Value Report (includes logical impossibilities):
{json.dumps(missing_report, indent=2)}

Recommend the best imputation strategy for each column listed above."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                json_mode=True,
                temperature=0.1,
            )
            result = json.loads(response)
            
            strats = result.get("strategies", [])
            for s in strats:
                await log(f"  [cyan]{s['column']}[/cyan] → [bold]{s['strategy']}[/bold] ({s['reasoning']})")
                
            return result
        except Exception as e:
            await log(f"  [bold red]⚠ Imputation Agent Error: {e}[/bold red]")
            return {"strategies": []}
