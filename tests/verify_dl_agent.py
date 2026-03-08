import asyncio
import pandas as pd
import numpy as np
from beyondml.agents.dl_agent import DeepLearningAgent
from beyondml.llm.ollama_provider import OllamaProvider

async def test_dl_agent():
    # 1. Create dummy tabular data
    data = {
        'feat1': np.random.rand(100),
        'feat2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    
    # 2. Mock log function
    async def mock_log(msg):
        print(f"LOG: {msg}")
        
    # 3. Instantiate Agent
    # We'll skip LLM for this test as the run method doesn't strictly need it for logic, 
    # but we'll provide a provider if needed.
    agent = DeepLearningAgent()
    
    # 4. Run tabular training
    print("Starting DL Agent verification...")
    try:
        result = await agent.run(
            df=df,
            target_column='target',
            problem_type='classification',
            log=mock_log,
            epochs=2
        )
        print("\nVerification Result:")
        print(f"Test Score: {result['test_score']}")
        print(f"Model Type: {result['model_type']}")
        
        if result['model_type'] == "SimpleMLP" and result['test_score'] >= 0:
            print("\n✅ Verification Successful!")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dl_agent())
