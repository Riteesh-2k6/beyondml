"""
Tests for the SHAP Explainability Agent.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from beyondml.agents.explainability_agent import ExplainabilityAgent, SHAP_AVAILABLE


@pytest.fixture
def mock_pipeline(classification_df):
    """Creates a simple fitted pipeline for explainability testing."""
    X = classification_df.drop(columns=["target", "category"])
    y = classification_df["target"]
    
    pipe = Pipeline([
        ("preprocessor", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe, X


@pytest.mark.asyncio
async def test_explainability_agent_run(mock_pipeline, mock_llm):
    """Test that the agent runs SHAP correctly and generates an explanation."""
    pipe, X_eval = mock_pipeline
    
    # We must skip the test if SHAP isn't available
    if not SHAP_AVAILABLE:
        pytest.skip("SHAP not available")
        
    async def mock_log(msg): pass
    
    agent = ExplainabilityAgent(mock_llm)
    result = await agent.run(
        model_pipeline=pipe,
        X_eval=X_eval,
        target_column="target",
        problem_type="classification",
        log=mock_log
    )
    
    print("REASON:", result.get("reason"))
    print("TRACEBACK:", result.get("traceback"))
    assert result["status"] == "success", f"Explainability agent failed: {result.get('reason')}"
    assert "top_features" in result
    assert "explanation" in result
    
    # Check that feature importance dict isn't empty and has expected columns
    top_features = result["top_features"]
    assert len(top_features) > 0
    assert any("age" in f or "income" in f or "score" in f for f in top_features.keys())
