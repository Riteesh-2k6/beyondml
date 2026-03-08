from typing import Dict, Any, List, Callable, Awaitable
from ..engine.deep_learning import SimpleCNN, SimpleMLP, DeepLearningTrainer
from ..engine.data_loader import load_mnist, load_cifar10, load_custom_images

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DeepLearningAgent:
    """Agent specialized in Deep Learning tasks."""
    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self.name = "Deep Learning Agent"

    async def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        log: Callable[[str], Awaitable[None]],
        epochs: int = 10
    ) -> Dict[str, Any]:
        """Main entry point for tabular DL training."""
        await log(f"\n[bold magenta]● {self.name}[/bold magenta] Starting Tabular DL Training...")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Simple encoding for target if categorical
        if problem_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            num_classes = len(le.classes_)
        else:
            num_classes = 1

        # Identify column types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        await log(f"  Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")
        
        # Build preprocessing pipeline (matches GA path quality)
        transformers = []
        if num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipe, num_cols))
        if cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, cat_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        X_processed = preprocessor.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        # Convert to Tensors
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)
        
        input_size = X_train.shape[1]
        model = SimpleMLP(input_size=input_size, num_classes=num_classes)
        trainer = DeepLearningTrainer(model)
        
        await log(f"  Training Simple MLP (Input: {input_size}, Classes: {num_classes})...")
        history = trainer.train(train_loader, test_loader, epochs=epochs)
        
        best_acc = max(history["test_acc"])
        await log(f"  [bold green]✓ DL Training Complete.[/bold green] Best Accuracy: {best_acc:.2f}%")
        
        return {
            "test_score": best_acc / 100.0,
            "history": history,
            "model_type": "SimpleMLP"
        }

    def run_mnist_demo(self, epochs: int = 5) -> Dict[str, Any]:
        """Orchestrate an MNIST training run."""
        print(f"[{self.name}] Starting MNIST demo...")
        train_loader, test_loader = load_mnist()
        model = SimpleCNN(num_classes=10, in_channels=1)
        trainer = DeepLearningTrainer(model)
        history = trainer.train(train_loader, test_loader, epochs=epochs)
        return {"dataset": "MNIST", "history": history}

    def run_cifar_demo(self, epochs: int = 5) -> Dict[str, Any]:
        """Orchestrate a CIFAR-10 training run."""
        print(f"[{self.name}] Starting CIFAR-10 demo...")
        train_loader, test_loader = load_cifar10()
        model = SimpleCNN(num_classes=10, in_channels=3)
        trainer = DeepLearningTrainer(model)
        history = trainer.train(train_loader, test_loader, epochs=epochs)
        return {"dataset": "CIFAR-10", "history": history}

    def train_on_custom_data(self, data_dir: str, epochs: int = 10) -> Dict[str, Any]:
        """Orchestrate training on a custom image directory."""
        print(f"[{self.name}] Loading custom data from {data_dir}...")
        train_loader, test_loader, class_map = load_custom_images(data_dir)
        num_classes = len(class_map)
        
        # Check first batch to get input channels
        first_batch = next(iter(train_loader))
        in_channels = first_batch[0].shape[1]
        
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
        trainer = DeepLearningTrainer(model)
        history = trainer.train(train_loader, test_loader, epochs=epochs)
        
        return {
            "dataset": "Custom",
            "data_dir": data_dir,
            "class_map": class_map,
            "history": history
        }
