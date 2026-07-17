import os
import subprocess
import pickle
import itertools as it
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / notebooks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


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

        # Both scripts have now pickled their test-set predictions to
        # results_dashboard/. Combine RA-GCN + the 5 traditional models into
        # one consolidated ROC image and one consolidated confusion-matrix
        # image for this dataset, instead of separate per-model figures.
        print(f"\nBuilding consolidated ROC and confusion matrix images for {name.upper()}")
        build_combined_figures(path)

    print("\n\n COMPLETED SUCCESSFULLY!")
    
    # Compile and display individual summary tables for each executed dataset
    os.makedirs("results_dashboard", exist_ok=True)
    for name in datasets.keys():
        matches = sorted(
            f for f in os.listdir("results_dashboard")
            if f.startswith(f"{name}_data_traditional_metrics_") and f.endswith(".csv")
        )
        if matches:
            csv_path = os.path.join("results_dashboard", matches[-1])  # most recent run
            print(f"\n\nFINAL REPORT SUMMARY TABLE ({name.upper()}):")
            df = pd.read_csv(csv_path)
            df = df[["Condition", "Model", "Accuracy", "Macro F1", "Binary F1", "ROC-AUC"]]
            print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# Consolidated figure builder (RA-GCN + 5 traditional models, per dataset)
# --------------------------------------------------------------------------- #
def build_combined_figures(path, output_dir="results_dashboard"):
    """
    Load RA-GCN's and the traditional baselines' pickled SMOTE-condition
    predictions for one dataset and produce exactly 2 images:
    one combined ROC plot and one combined confusion-matrix grid,
    each showing all 6 models (RA-GCN + 5 traditional) together.
    """
    dataset_name = os.path.splitext(os.path.basename(path))[0]

    ragcn_path = os.path.join(output_dir, f"{dataset_name}_ragcn_predictions.pkl")
    trad_path = os.path.join(output_dir, f"{dataset_name}_traditional_predictions.pkl")

    if not (os.path.exists(ragcn_path) and os.path.exists(trad_path)):
        print(f"\nSkipping combined figures for {dataset_name}: prediction files not found.")
        return

    with open(ragcn_path, "rb") as f:
        ragcn_preds = pickle.load(f)
    with open(trad_path, "rb") as f:
        trad_preds = pickle.load(f)

    # RA-GCN first, then the 5 traditional models, so it's visually anchored
    # as the reference model in both the legend and the grid.
    all_models = {"RA-GCN": ragcn_preds}
    all_models.update(trad_preds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _plot_combined_roc(dataset_name, all_models, output_dir, timestamp)
    _plot_combined_confusion_matrices(dataset_name, all_models, output_dir, timestamp)


def _plot_combined_roc(dataset_name, all_models, output_dir, timestamp):
    """One ROC plot, 6 curves (RA-GCN + 5 traditional), AUC-only legend for readability."""
    plt.figure(figsize=(8.5, 6.5), dpi=300)
    colors = ["#e41a1c", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    for (model_name, preds), color in zip(all_models.items(), colors):
        y_true, y_score = preds["y_true"], preds["y_score"]
        auc_val = preds.get("auc", roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{model_name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves \u2013 RA-GCN vs Traditional Baselines\nDataset: {dataset_name.upper()}", fontsize=13, pad=10)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_combined_roc_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Combined ROC Saved] -> {out_path}")


def _plot_combined_confusion_matrices(dataset_name, all_models, output_dir, timestamp):
    """One image, 2x3 grid of confusion-matrix heatmaps, one panel per model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    classes = ["Class 0", "Class 1"]

    for ax, (model_name, preds) in zip(axes, all_models.items()):
        cm = confusion_matrix(preds["y_true"], preds["y_pred"])
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(model_name, fontsize=12, weight="bold")
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks); ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=9)

        thresh = cm.max() / 2.
        for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], "d"), ha="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=11, weight="bold")
        ax.set_ylabel("True", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)

    fig.suptitle(f"Confusion Matrices \u2013 RA-GCN vs Traditional Baselines\nDataset: {dataset_name.upper()}",
                 fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_combined_confusion_matrix_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Combined Confusion Matrix Saved] -> {out_path}")


if __name__ == "__main__":
    run_pipeline()
