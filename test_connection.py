"""
BeyondML LLM Connection Tester
Utility to verify Ollama or Groq connectivity.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from beyondml.llm import get_llm_provider

def main():
    print("🔍 BeyondML LLM Connection Tester")
    print("=" * 40)
    
    try:
        provider = get_llm_provider()
        name = provider.model_name
        print(f"📡 Provider: {name}")
        
        print(f"🔄 Testing connection to {name}...")
        is_connected = provider.test_connection()
        
        if is_connected:
            print(f"✅ SUCCESS: Connected to {name}!")
            
            print("\n💬 Sending test message...")
            try:
                response = provider.chat([{"role": "user", "content": "Say 'Connection established!'" }], timeout=30)
                print(f"🤖 Response: {response.strip()}")
                print("\n✨ Everything looks good!")
            except Exception as e:
                print(f"❌ Chat failed: {e}")
        else:
            print(f"❌ FAILED: Could not reach {name}.")
            if "ollama" in name.lower():
                print("\n💡 Troubleshooting Ollama:")
                print("  1. Is Ollama running? (Try 'ollama serve')")
                print(f"  2. Is the model '{name.split('/')[-1]}' pulled? (Try 'ollama pull {name.split('/')[-1]}')")
                print("  3. Is OLLAMA_HOST correct? (Default: http://localhost:11434)")
            elif "groq" in name.lower():
                print("\n💡 Troubleshooting Groq:")
                print("  1. Is GROQ_API_KEY set correctly in .env?")
                print("  2. Do you have internet access?")
                
    except Exception as e:
        print(f"💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
