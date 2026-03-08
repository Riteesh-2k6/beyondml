"""
Orchestrator Agent — LLM-powered path router.

Sends dataset summary to Groq, gets back a routing decision with reasoning.
User can override the AI's recommendation at every decision point.
"""

import json
from typing import Dict, Any, Callable, Awaitable, Optional
from ..llm.base import LLMProvider


SYSTEM_PROMPT = """You are an expert ML engineer acting as an orchestrator for an AutoML platform.
Given a dataset summary (which includes column names, types, and sample values), you must decide the best analysis path.

You MUST respond with a JSON object containing:
{
  "path": "supervised" or "unsupervised" or "explore" or "deep_learning",
  "reasoning": "2-3 sentence explanation of why this path is best. Explain your logic.",
  "suggested_target": "exact_column_name_from_dataset" or null if unsupervised,
  "confidence": "high" or "medium" or "low",
  "task_type": "classification" or "regression" or "clustering" or "exploration",
  "model_recommendations": ["model1", "model2"]
}

Rules:
1. Target Selection: A target should be what the user most likely wants to predict based on the dataset's nature.
   - Look for columns representing outcomes, labels, categories, prices, statuses, or boolean flags.
   - Look at the "Sample Values" carefully. If a column contains mostly unique text, dates, UUIDs, or monotonically increasing IDs, it is NOT a target.
2. Unsupervised Path: This is CRITICAL. If the dataset looks like raw logs, transactional events, OR consists entirely of aggregate metrics (e.g., columns starting or ending with 'total', 'pending', 'disposed', 'amount', 'count'), WITHOUT a clear, single classification label, you MUST confidently route to 'unsupervised' and set "suggested_target" to null. Do not force an arbitrary column to be a target if they are all interchangeable tracking metrics.
3. Supervised Path: If there IS a clear, singular target variable (labeled data) to predict, choose "supervised".
4. Deep Learning: If the data is highly complex, non-linear, or image-based, consider "deep_learning".
5. Explore: If the user explicitly asks just to explore or understand their data, choose "explore".
"""


class OrchestratorAgent:
    """Routes the pipeline based on LLM analysis + user override."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df_summary: str,
        description: str,
        user_target: Optional[str],
        user_path_choice: Optional[str],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        """
        Analyze dataset and decide the pipeline path.

        Args:
            df_summary: String summary of DataFrame (shape, dtypes, sample rows, stats)
            description: User's natural language description of the dataset
            user_target: Optional explicit target defined by the user
            user_path_choice: User's explicit path choice (if any), None for auto
            log: Async callback to write to TUI log
        """
        await log("[bold magenta]● Orchestrator[/bold magenta]  Analyzing dataset...")

        # If user already chose a specific path, respect that
        if user_path_choice and user_path_choice not in ["auto", "autonomous"]:
            path_map = {
                "explore": "explore",
                "supervised": "supervised",
                "unsupervised": "unsupervised",
                "deep_learning": "deep_learning",
                "dimensionality_reduction": "dimensionality_reduction"
            }
            path = path_map.get(user_path_choice, "supervised")
            await log(f"  User selected path: [bold cyan]{path}[/bold cyan]")

            # Still ask LLM for target/model suggestions
            result = await self._ask_llm(df_summary, description, user_target, log)
            result["path"] = path

            # Adjust task_type based on user's path choice
            if path == "unsupervised":
                result["task_type"] = "clustering"
                result["suggested_target"] = None
            elif path in ["explore", "dimensionality_reduction"]:
                result["task_type"] = "exploration"

            return result

        # Auto mode: let LLM decide
        result = await self._ask_llm(df_summary, description, user_target, log)
        return result

    async def _ask_llm(
        self,
        df_summary: str,
        description: str,
        user_target: Optional[str],
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        user_target_str = f"User explicitly suggests evaluating this target: {user_target}" if user_target else "No explicit target provided by user."
        
        user_msg = f"""Dataset Summary:
{df_summary}

User Description: {description or 'No description provided'}

User Preference:
- {user_target_str}

Analyze this dataset and decide the best analysis path. Provide the JSON output."""

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
            await log("  Falling back to basic exploration...")

            # Fallback to pure exploration if LLM fails completely
            return {
                "path": "explore",
                "reasoning": f"Hard fallback due to LLM failure: {e}",
                "suggested_target": None,
                "confidence": "low",
                "task_type": "exploration",
                "model_recommendations": [],
            }
