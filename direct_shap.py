import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def classification_df():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.normal(50000, 15000, n).round(2),
        "score": rng.uniform(0, 100, n).round(2),
        "target": rng.choice([0, 1], n),
    })
    return df

df = classification_df()
X = df.drop(columns=['target'])
y = df['target']
pipe = Pipeline([
    ('preprocessor', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10, random_state=42))
])
pipe.fit(X, y)

X_eval = X
preprocessor = pipe.named_steps["preprocessor"]
model = pipe.named_steps["model"]
X_transformed = preprocessor.transform(X_eval)
feature_names = preprocessor.get_feature_names_out()
feature_names = [f.split("__")[-1] for f in feature_names]

sample_size = min(100, X_transformed.shape[0])
idx = np.random.choice(X_transformed.shape[0], sample_size, replace=False)
X_sample = X_transformed[idx]

print("Running TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("shap_values shape:", np.shape(shap_values))

if isinstance(shap_values, list):
    print("List of shapes:", [s.shape for s in shap_values])
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

global_shap = np.abs(shap_values).mean(axis=0)

importance_dict = {name: float(val) for name, val in zip(feature_names, global_shap)}
print(importance_dict)
