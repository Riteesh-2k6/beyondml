import pytest
import pandas as pd
import numpy as np
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.agents.eda_agent import EDAAgent

@pytest.mark.asyncio
async def test_ga_progress_graph_accuracy(classification_df, mock_llm):
    """Verifies that the generational progress graph tracks correctly."""
    ga_agent = GATrainerAgent(mock_llm)
    
    async def mock_get_input(prompt): return ""
    async def mock_log(msg): pass
    
    from beyondml.engine.profiler import DatasetProfiler
    profiler = DatasetProfiler(classification_df, target_column="target")
    profile = profiler.run()
    
    # Run GA for 3 generations
    result = await ga_agent.run(
        df=classification_df,
        target_column="target",
        profile=profile,
        model_choice="RandomForest",
        log=mock_log,
        get_user_input=mock_get_input,
        pop_size=5,
        generations=3
    )
    
    history = result["ga_history"]
    assert len(history) >= 2 # Should have at least gen 0 and gen 1
    
    # Check monotonicity of best fitness (best fitness should stay same or increase)
    best_fitnesses = [h["best_fitness"] for h in history]
    for i in range(1, len(best_fitnesses)):
        assert best_fitnesses[i] >= best_fitnesses[i-1], f"Fitness regression at gen {i}"

@pytest.mark.asyncio
async def test_eda_chart_rendering_accuracy(classification_df, mock_llm):
    """Verifies that EDA agent renders non-empty charts."""
    eda_agent = EDAAgent(mock_llm)
    
    async def mock_log(msg): pass
    
    from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
    profiler = DatasetProfiler(classification_df)
    profile = profiler.run()
    target_id = TargetIdentifier(classification_df)
    target_info = target_id.identify()
    
    # Mock LLM to recommend a histogram and scatter
    import json
    mock_llm.responses["default"] = json.dumps({
        "insights": [],
        "chart_recs": [
            {"type": "histogram", "columns": ["age"], "rationale": "test"},
            {"type": "scatter", "columns": ["age", "income"], "rationale": "test"}
        ],
        "suggested_target": "target",
        "target_confidence": 0.9,
        "task_type": "classification",
        "narrative": "test"
    })
    
    result = await eda_agent.run(
        df=classification_df,
        profile=profile,
        target_info=target_info,
        description="test",
        log=mock_log
    )
    
    charts = result["rendered_charts"]
    assert len(charts) >= 1
    for title, chart_str in charts:
        assert isinstance(chart_str, str)
        assert len(chart_str) > 0 # ANSI/Plotext output should be non-empty
