import json
from typing import Dict, Any, List, Callable, Awaitable
from ..llm.base import LLMProvider

SYSTEM_PROMPT = """You are a Data Sanity Expert. Your role is to identify logical impossibilities or common fallback values (semantic nulls) in a dataset based on its domain.

Given a list of columns, their descriptions, and their statistical distributions (min, max, median, unique counts), identify values that are logically or physically impossible given the context.

Common examples:
- Blood Pressure = 0 (Biological impossibility, likely a missing value)
- Age = 999 (Sentinel value for missing)
- Year = 2050 (Future year in a historical dataset)
- Interest Rate = -5 (Impossible in standard finance)
- Zip Code = 00000 (Placeholder)

You MUST respond with a JSON object:
{
  "issues": [
    {
      "column": "column_name",
      "invalid_values": [value1, value2],
      "reasoning": "Explanation of why these are impossible"
    }
  ]
}

If no issues are found, return an empty list for "issues".
Only flag values that are CLEARLY impossible, not just rare (outliers).
"""

class SanityAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df_summary: str,
        column_stats: Dict[str, Any],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log("[bold yellow]● Sanity Agent[/bold yellow]  Auditing logical distributions...")
        
        user_msg = f"""Dataset Context:
{df_summary}

Column Statistics:
{json.dumps(column_stats, indent=2)}

Audit this data for logical impossibilities."""

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
            
            issues = result.get("issues", [])
            if issues:
                for issue in issues:
                    await log(f"  [red]⚠ {issue['column']}[/red]: Found invalid values {issue['invalid_values']} ({issue['reasoning']})")
            else:
                await log("  [green]✓ No logical impossibilities detected.[/green]")
                
            return result
        except Exception as e:
            await log(f"  [bold red]⚠ Sanity Agent Error: {e}[/bold red]")
            return {"issues": []}
