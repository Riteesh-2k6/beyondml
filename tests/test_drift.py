"""
Tests for Data Drift Agent.
"""
import pytest
import pandas as pd
import numpy as np
from beyondml.agents.drift_agent import DriftAgent


@pytest.fixture
def reference_df():
    """Reference dataset (e.g. training data)."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "num_stable": rng.normal(0, 1, n),
        "num_drifted": rng.normal(0, 1, n),
        "cat_stable": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "cat_drifted": rng.choice(["X", "Y"], n, p=[0.8, 0.2]),
    })
    return df


@pytest.fixture
def current_df():
    """Current dataset (e.g. inference data)."""
    rng = np.random.RandomState(45)
    n = 200
    df = pd.DataFrame({
        "num_stable": rng.normal(0, 1, n),  # stable distribution
        "num_drifted": rng.normal(2.5, 1, n),  # drifted distribution (shifted mean)
        "cat_stable": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]), # stable
        "cat_drifted": rng.choice(["X", "Y"], n, p=[0.1, 0.9]), # drifted
    })
    return df


@pytest.mark.asyncio
async def test_drift_agent_detection(reference_df, current_df, mock_llm):
    """Test that DriftAgent correctly identifies drifted columns."""
    async def mock_log(msg): pass
    
    agent = DriftAgent(mock_llm)
    result = await agent.run(reference_df, current_df, mock_log, p_value_threshold=0.05)
    
    assert result["status"] == "success"
    drifting_features = result["drifting_features"]
    
    assert "num_drifted" in drifting_features
    assert "cat_drifted" in drifting_features
    assert "num_stable" not in drifting_features
    assert "cat_stable" not in drifting_features
    
    assert "drift_narrative" in result

@pytest.mark.asyncio
async def test_drift_agent_no_common_columns(mock_llm):
    """Test that DriftAgent handles missing common columns gracefully."""
    df_ref = pd.DataFrame({"A": [1, 2]})
    df_cur = pd.DataFrame({"B": [1, 2]})
    
    async def mock_log(msg): pass
    agent = DriftAgent(mock_llm)
    result = await agent.run(df_ref, df_cur, mock_log)
    
    assert result["status"] == "error"
    assert result["reason"] == "No common columns"
