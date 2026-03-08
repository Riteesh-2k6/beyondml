import pytest
import json
import pandas as pd
import numpy as np
from beyondml.agents.reflection_agent import ReflectionAgent
from beyondml.agents.dl_agent import DeepLearningAgent

@pytest.mark.asyncio
async def test_reflection_agent_satisfied(mock_llm):
    agent = ReflectionAgent(mock_llm)
    eval_result = {
        "test_score": 0.95,
        "train_score": 0.96,
        "feature_importances": {"a": 0.2, "b": 0.3}
    }
    
    async def mock_log(msg): pass
    
    result = await agent.run(eval_result, iteration=1, max_iterations=3, log=mock_log)
    assert result["status"] == "satisfied"
    assert result["modifications"] is None

@pytest.mark.asyncio
async def test_reflection_agent_needs_improvement(mock_llm):
    # Setup mock response for overfitting
    mock_llm.responses["default"] = json.dumps({
        "reasoning": "Overfitting detected.",
        "features_to_drop": ["highly_correlated_col"],
        "new_features": [{"name": "ratio", "expr": "df['a']/df['b']", "rationale": "Better signal"}],
        "next_model": "RandomForest",
        "next_ga_generations": 10,
        "next_ga_pop_size": 20
    })
    
    agent = ReflectionAgent(mock_llm)
    # Trigger overfitting rule: train_score - test_score > 0.10
    eval_result = {
        "test_score": 0.70,
        "train_score": 0.90,
        "feature_importances": {"a": 0.1, "b": 0.1}
    }
    
    async def mock_log(msg): pass
    
    result = await agent.run(eval_result, iteration=1, max_iterations=3, log=mock_log)
    assert result["status"] == "needs_improvement"
    assert "highly_correlated_col" in result["modifications"]["features_to_drop"]
    assert result["modifications"]["next_model"] == "RandomForest"

@pytest.mark.asyncio
async def test_dl_agent_tabular_run(classification_df):
    agent = DeepLearningAgent()
    
    async def mock_log(msg): print(msg)
    
    # We use a small number of epochs for the test
    result = await agent.run(
        df=classification_df,
        target_column="target",
        problem_type="classification",
        log=mock_log,
        epochs=1
    )
    
    assert "test_score" in result
    assert result["model_type"] == "SimpleMLP"
    assert result["test_score"] >= 0
