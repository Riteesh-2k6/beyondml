"""
Orchestrator Agent — LLM-powered path router.

Sends dataset summary to Groq, gets back a routing decision with reasoning.
User can override the AI's recommendation at every decision point.
"""

import json
from typing import Dict, Any, Callable, Awaitable, Optional
from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are an expert ML engineer acting as an orchestrator for an AutoML platform.
Given a dataset summary, you must decide the best analysis path.

You MUST respond with a JSON object containing:
{
  "path": "supervised" or "unsupervised" or "explore",
  "reasoning": "2-3 sentence explanation of why this path is best",
  "suggested_target": "column_name or null if unsupervised",
  "confidence": "high" or "medium" or "low",
  "task_type": "classification" or "regression" or "clustering" or "exploration",
  "model_recommendations": ["model1", "model2"]
}

Rules:
- If there's a clear target variable (labeled data), choose "supervised"
- If there's no clear target and the data has many features, choose "unsupervised"
- If the user just wants to understand their data, choose "explore"
- Always suggest the most appropriate target column for supervised tasks
- Recommend 2-3 model types that would work well
"""


class OrchestratorAgent:
    """Routes the pipeline based on LLM analysis + user override."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df_summary: str,
        description: str,
        target_info: Dict[str, Any],
        user_path_choice: Optional[str],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        """
        Analyze dataset and decide the pipeline path.

        Args:
            df_summary: String summary of DataFrame (shape, dtypes, sample rows, stats)
            description: User's natural language description of the dataset
            target_info: Output from TargetIdentifier
            user_path_choice: User's explicit path choice (if any), None for auto
            log: Async callback to write to TUI log
        """
        await log("[bold magenta]● Orchestrator[/bold magenta]  Analyzing dataset...")

        # If user already chose a specific path, respect that
        if user_path_choice and user_path_choice != "auto":
            path_map = {
                "explore": "explore",
                "supervised": "supervised",
                "unsupervised": "unsupervised",
            }
            path = path_map.get(user_path_choice, "supervised")
            await log(f"  User selected path: [bold cyan]{path}[/bold cyan]")

            # Still ask LLM for target/model suggestions
            result = await self._ask_llm(df_summary, description, target_info, log)
            result["path"] = path

            # Adjust task_type based on user's path choice
            if path == "unsupervised":
                result["task_type"] = "clustering"
                result["suggested_target"] = None
            elif path == "explore":
                result["task_type"] = "exploration"

            return result

        # Auto mode: let LLM decide
        result = await self._ask_llm(df_summary, description, target_info, log)
        return result

    async def _ask_llm(
        self,
        df_summary: str,
        description: str,
        target_info: Dict[str, Any],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        user_msg = f"""Dataset Summary:
{df_summary}

User Description: {description or 'No description provided'}

Deterministic Target Analysis:
- Suggested target: {target_info.get('suggested_target', 'None')}
- Confidence: {target_info.get('confidence_score', 0):.2f}
- Top candidates: {target_info.get('ranked_candidates', [])[:5]}

Analyze this dataset and decide the best analysis path."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                json_mode=True,
                temperature=0.3,
            )
            result = json.loads(response)

            await log(f"  Path: [bold green]{result.get('path', 'supervised')}[/bold green]")
            await log(f"  [bold cyan]Reasoning:[/bold cyan]")
            await log(f"    [dim]{result.get('reasoning', 'N/A')}[/dim]")
            if result.get("suggested_target"):
                await log(f"  Suggested target: [bold yellow]{result['suggested_target']}[/bold yellow]")
            await log(f"  Confidence: {result.get('confidence', 'medium')}")
            if result.get("model_recommendations"):
                await log(f"  Recommended models: {', '.join(result['model_recommendations'])}")

            return result

        except Exception as e:
            await log(f"  [bold red]⚠ LLM error: {e}[/bold red]")
            await log("  Falling back to deterministic analysis...")

            # Fallback to deterministic logic
            confidence = target_info.get("confidence_score", 0)
            if confidence > 0.3:
                path = "supervised"
                task_type = "classification"
            else:
                path = "unsupervised"
                task_type = "clustering"

            return {
                "path": path,
                "reasoning": f"Deterministic fallback (LLM unavailable). Target confidence: {confidence:.2f}",
                "suggested_target": target_info.get("suggested_target"),
                "confidence": "low",
                "task_type": task_type,
                "model_recommendations": ["RandomForest"],
            }
