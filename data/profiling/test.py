import pandas as pd
import json

from profiler import DatasetProfiler, TargetIdentifier

file_path = "C:\Programming\Hackveda\dataiku-dss-13.3.2\python36.packages\sklearn\datasets\data\iris.csv"
def test_custom_dataset(file_path):
    print(f"\nLoading dataset from: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)


    print("\nDataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Step 1: Identify Target Automatically
    identifier = TargetIdentifier(df)
    target_info = identifier.identify()

    print("\n=== Target Identification ===")
    print("Suggested Target:", target_info["suggested_target"])
    print("Confidence Score:", target_info["confidence_score"])
    print("Top 5 Ranked Candidates:", target_info["ranked_candidates"][:5])

    # Optional: Accept auto target if confidence high
    if target_info["confidence_score"] > 0.6:
        target_column = target_info["suggested_target"]
        print("\nAuto-accepting target:", target_column)
    else:
        print("\nLow confidence. Please verify target manually.")
        return

    # Step 2: Run Full Profiler
    profiler = DatasetProfiler(df, target_column=target_column)
    profile = profiler.run()

    print("\n=== Metadata ===")
    print(json.dumps(profile["metadata"], indent=2))

    print("\n=== Target Analysis ===")
    print(json.dumps(profile["target_analysis"], indent=2))

    print("\n=== Feature Types ===")
    print(json.dumps(profile["feature_types"], indent=2))

    print("\n=== Missing Analysis (first 5 columns) ===")
    missing = profile["missing_analysis"]["missing_percentage"]
    print(dict(list(missing.items())[:5]))
    print(df.head())


if __name__ == "__main__":
    # Replace with your dataset path
    test_custom_dataset(file_path)
