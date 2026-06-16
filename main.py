import os
import subprocess
import pandas as pd

def run_pipeline():
    # Explicitly mapping out all 5 clinical benchmark datasets
    datasets = {
        "oasis": "data/oasis/oasis_data.pkl",
        "diabetes": "data/diabetes/diabetes_data.pkl",
        "parkinsons": "data/parkinsons/parkinsons_data.pkl",
        "heart": "data/heart/heart_data.pkl",
        "maternal": "data/maternal/maternal_data.pkl"
    }

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"\nSkipping: Dataset file not found at {path}")
            continue

        print(f"\n\nExecuting benchmarking suite for dataset: {name.upper()}")
        
        print(f"\nLaunching RA-GCN Model via main_medical.py")
        cmd_gcn = ["python", "code/main_medical.py", "--dataset", path]
        subprocess.run(cmd_gcn, check=True)

        print(f"\nLaunching 5 Traditional Baselines via traditional_baselines.py")
        cmd_trad = ["python", "code/traditional_baselines.py", "--dataset", path]
        subprocess.run(cmd_trad, check=True)

    print("\n\n COMPLETED SUCCESSFULLY!")
    
    # Compile and display individual summary tables for each executed dataset
    for name in datasets.keys():
        csv_path = f"{name}_traditional_metrics.csv"
        if os.path.exists(csv_path):
            print(f"\n\nFINAL REPORT SUMMARY TABLE ({name.upper()}):")
            df = pd.read_csv(csv_path)
            df = df[["Condition", "Model", "Accuracy", "Macro F1", "Binary F1", "ROC-AUC"]]
            print(df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
