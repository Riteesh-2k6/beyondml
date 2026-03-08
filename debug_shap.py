import asyncio
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from beyondml.agents.explainability_agent import ExplainabilityAgent
from tests.conftest import classification_df, mock_llm

async def run():
    logging.basicConfig(level=logging.DEBUG)
    df = classification_df()
    from tests.conftest import MockLLM
    llm = MockLLM()
    X = df.drop(columns=['target'])
    y = df['target']
    
    pipe = Pipeline([
        ('preprocessor', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipe.fit(X, y)
    
    agent = ExplainabilityAgent(llm)
    async def log(msg): print(msg)
    
    result = await agent.run(pipe, X, 'target', 'classification', log)
    print("RESULT:", result)

if __name__ == "__main__":
    asyncio.run(run())
