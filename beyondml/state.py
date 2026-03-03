"""
MLState — the single source of truth flowing through every agent.
"""

from typing import TypedDict, Optional, Literal, List, Dict, Any
import pandas as pd


class MLState(TypedDict, total=False):
    # ── Data & Entry
    dataset_path: str
    description: str
    df: pd.DataFrame
    df_original: pd.DataFrame
    path: Literal["explore", "supervised", "unsupervised"]

    # ── EDA Results
    eda_insights: List[Dict[str, Any]]
    chart_recs: List[Dict[str, Any]]
    rendered_charts: List[str]
    null_strategy: Dict[str, str]
    outlier_indices: List[Dict[str, Any]]
    outlier_columns: List[str]
    suggested_target: str
    confirmed_target: Optional[str]

    # ── Feature Engineering
    feature_proposals: List[Dict[str, Any]]
    features_applied: List[str]
    features_rejected: List[str]
    outlier_strategy: str

    # ── GA Training
    ga_config: Dict[str, Any]
    ga_history: List[Dict[str, Any]]
    best_params: Dict[str, Any]
    best_cv_score: float
    model: object
    model_type: str

    # ── Evaluation & Output
    test_score: float
    eval_report: Dict[str, Any]
    feature_importances: Dict[str, float]
    eval_narration: str
    model_path: str

    # ── Control Flow & TUI
    current_node: str
    awaiting_input: bool
    input_prompt: str
    user_response: Optional[str]
    messages: List[Dict[str, str]]
    errors: List[str]
