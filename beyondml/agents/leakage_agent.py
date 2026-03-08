import json
from typing import Dict, Any, List, Callable, Awaitable
from ..llm.base import LLMProvider

SYSTEM_PROMPT = """You are a Data Leakage Expert. Your role is to identify features that "leak" information about the target variable.

Data leakage occurs when a feature contains information that wouldn't be available at the time of prediction, or is essentially a proxy for the target.

Given the target column, column descriptions, and correlations with the target, identify likely leakage.

Examples:
- Target: "Churn", Feature: "Cancellation_Date" (Leak: Date is only set AFTER churn)
- Target: "Disease", Feature: "Prescription_A" (Leak: Treatment implies diagnosis)
- Target: "Price", Feature: "Tax_Amount" (Leak: Tax is a direct function of Price)

You MUST respond with a JSON object:
{
  "leakage_findings": [
    {
      "column": "column_name",
      "risk_level": "high" or "medium",
      "reasoning": "Why this is considered leakage"
    }
  ],
  "recommendations": [
    {
      "column": "column_name",
      "action": "drop" or "monitor",
      "rationale": "Why drop or monitor"
    }
  ]
}

If no leakage is found, return empty lists.
"""

class LeakageAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        target_column: str,
        column_descriptions: str,
        correlations: Dict[str, float],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log("[bold cyan]● Leakage Agent[/bold cyan]  Auditing for target leakage...")
        
        user_msg = f"""Target Column: {target_column}
        
Column Descriptions:
{column_descriptions}

Correlations with Target:
{json.dumps(correlations, indent=2)}

Identify any columns that might be leaking target information."""

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
            
            findings = result.get("leakage_findings", [])
            if findings:
                for find in findings:
                    await log(f"  [red]⚠ Leakage Risk[/red]: [bold]{find['column']}[/bold] ({find['risk_level']}) - {find['reasoning']}")
            else:
                await log("  [green]✓ No obvious data leakage detected.[/green]")
                
            return result
        except Exception as e:
            await log(f"  [bold red]⚠ Leakage Agent Error: {e}[/bold red]")
            return {"leakage_findings": [], "recommendations": []}
