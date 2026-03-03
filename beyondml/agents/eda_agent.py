"""
EDA Agent — LLM-powered exploratory data analysis.

Runs profiler, asks Groq for insights and chart recommendations,
renders plotext charts, detects outliers.
"""

import json
from typing import Dict, Any, Callable, Awaitable, List
import pandas as pd
import numpy as np

from ..llm.base import LLMProvider
from ..engine.profiler import DatasetProfiler, TargetIdentifier
from ..charts import render_histogram, render_scatter, render_correlation_matrix, render_box_plot


SYSTEM_PROMPT = """You are a senior data scientist performing exploratory data analysis.
Given a dataset summary with statistics, you must provide structured findings.

You MUST respond with a JSON object containing:
{
  "insights": [
    {"finding": "description of insight", "severity": "low|medium|high"}
  ],
  "chart_recs": [
    {"type": "histogram|scatter|box|bar", "columns": ["col1", "col2"], "rationale": "why this chart"}
  ],
  "suggested_target": "column_name or null",
  "target_confidence": 0.0 to 1.0,
  "task_type": "classification|regression|clustering",
  "outlier_columns": ["columns with notable outliers"],
  "null_strategy": {"column_name": "mean|median|mode|drop"},
  "narrative": "2-3 sentence natural language summary of the dataset"
}

Provide 3-5 key insights and 2-4 chart recommendations.
Focus on actionable findings that affect modeling decisions.
"""


class EDAAgent:
    """LLM-powered EDA with chart rendering and outlier detection."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target_info: Dict[str, Any],
        description: str,
        log: Callable[[str], Awaitable[None]],
    ) -> Dict[str, Any]:
        await log(f"[bold cyan]● EDA Agent[/bold cyan]  Analysing {df.shape[0]}×{df.shape[1]} dataset...")

        # Build summary for LLM
        summary = self._build_summary(df, profile, target_info)

        # Ask LLM for insights
        eda_result = await self._ask_llm(summary, description, log)

        # Log insights
        insights = eda_result.get("insights", [])
        for insight in insights:
            severity = insight.get("severity", "low")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🟢")
            await log(f"  {icon} {insight.get('finding', '')}")

        # Render charts
        rendered_charts = []
        chart_recs = eda_result.get("chart_recs", [])
        num_cols = profile["feature_types"]["numerical"]

        # Always render correlation matrix if enough numeric columns
        if len(num_cols) >= 2:
            corr_str = render_correlation_matrix(profile.get("correlation_matrix", {}))
            rendered_charts.append(("Correlation Matrix", corr_str))
            await log(f"\n  [bold magenta]📊 CORRELATION: [/bold magenta][italic]{eda_result.get('suggested_target', 'None')}[/italic]")

        for rec in chart_recs[:4]:
            chart_type = rec.get("type", "histogram")
            columns = rec.get("columns", [])
            try:
                if chart_type == "histogram" and columns:
                    col = columns[0]
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        chart_str = render_histogram(df[col], title=f"Distribution of {col}")
                        rendered_charts.append((f"Histogram: {col}", chart_str))
                elif chart_type == "scatter" and len(columns) >= 2:
                    c1, c2 = columns[0], columns[1]
                    if c1 in df.columns and c2 in df.columns:
                        chart_str = render_scatter(df[c1], df[c2], title=f"Scatter: {c1} vs {c2}")
                        rendered_charts.append((f"Scatter: {c1} vs {c2}", chart_str))
                elif chart_type == "box" and columns:
                    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                    if valid_cols:
                        chart_str = render_box_plot(df, valid_cols, title="Box Plots")
                        rendered_charts.append(("Box Plots", chart_str))
            except Exception as e:
                await log(f"  [dim]⚠ Chart render failed: {e}[/dim]")

        if rendered_charts:
            await log(f"\n  [green]✓ Rendered {len(rendered_charts)} charts[/green]")

        # Log target suggestion
        target = eda_result.get("suggested_target")
        if target:
            conf = eda_result.get("target_confidence", 0)
            await log(f"\n  Suggested target: [bold green]{target}[/bold green] (confidence: {conf:.2f})")

        # Narrative
        narrative = eda_result.get("narrative", "")
        if narrative:
            await log(f"\n  [italic dim]{narrative}[/italic dim]")

        return {
            "eda_insights": insights,
            "chart_recs": chart_recs,
            "rendered_charts": rendered_charts,
            "suggested_target": target,
            "target_confidence": eda_result.get("target_confidence", 0),
            "task_type": eda_result.get("task_type", "classification"),
            "outlier_columns": eda_result.get("outlier_columns", []),
            "null_strategy": eda_result.get("null_strategy", {}),
            "model_recommendations": eda_result.get("model_recommendations", []),
        }

    def _build_summary(self, df: pd.DataFrame, profile: Dict, target_info: Dict) -> str:
        lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            f"Dtypes: {df.dtypes.to_dict()}",
            f"\nFirst 3 rows:\n{df.head(3).to_string()}",
            f"\nNumerical summary:\n{json.dumps(profile.get('numerical_summary', {}), indent=2)[:1500]}",
            f"\nMissing: {profile.get('missing_analysis', {})}",
            f"\nFeature types: {profile.get('feature_types', {})}",
            f"\nOutlier summary: {json.dumps(profile.get('outlier_summary', {}), indent=2)[:800]}",
            f"\nTarget analysis: {target_info}",
        ]
        return "\n".join(lines)

    async def _ask_llm(self, summary: str, description: str, log) -> Dict[str, Any]:
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Dataset description: {description or 'None'}\n\n{summary}"},
                ],
                json_mode=True,
                temperature=0.3,
            )
            return json.loads(response)
        except Exception as e:
            await log(f"  [bold red]⚠ EDA LLM error: {e}[/bold red]")
            return {
                "insights": [{"finding": "LLM unavailable, using basic profiling", "severity": "medium"}],
                "chart_recs": [],
                "suggested_target": None,
                "target_confidence": 0,
                "task_type": "classification",
                "outlier_columns": [],
                "null_strategy": {},
                "narrative": "Basic profiling completed without LLM insights.",
            }
