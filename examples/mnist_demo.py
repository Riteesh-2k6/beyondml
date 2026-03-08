"""
BeyondML MNIST Demo: Solving the digits classification task.
"""

from beyondml.agents.dl_agent import DeepLearningAgent

def main():
    print("=== BeyondML Deep Learning Demo ===")
    
    # Initialize the DL Agent
    agent = DeepLearningAgent()
    
    # Run the MNIST training demo for 3 epochs
    # This will:
    # 1. Download/Load MNIST images
    # 2. Build a SimpleCNN model
    # 3. Train and evaluate the model
    # 4. Print progress to terminal
    results = agent.run_mnist_demo(epochs=3)
    
    print("\nTraining Complete!")
    print(f"Dataset: {results['dataset']}")
    final_acc = results['history']['test_acc'][-1]
    print(f"Final Test Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    main()
