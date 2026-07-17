"""
traditional_baselines.py

Benchmarks five traditional ML models (Decision Tree, Random Forest, KNN,
Logistic Regression, ANN/MLP) against the RA-GCN clinical datasets, both on
raw imbalanced data and on SMOTE-rebalanced data.

It reuses `load_data_medical` from utils.py so every baseline sees the exact
same patient-level idx_train split used by RA-GCN itself, keeping the comparison fair.

Each model's SMOTE-condition test predictions are pickled to disk, and main.py later
combines them with RA-GCN's own predictions into two consolidated per-dataset
images (one shared ROC plot, one shared confusion-matrix grid) covering all
6 models together.
"""

import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE

# Make sure utils.py (sitting next to this script in code/) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data_medical  # custom RA-GCN data loader

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def resolve_dataset_path(path):
    """Resolve a dataset path whether the script is run from repo root or code/."""
    if os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


def to_numpy(x):
    """Convert torch tensors / lists / arrays returned by load_data_medical to numpy."""
    if hasattr(x, "detach"):       # torch tensor with autograd
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):        # plain torch tensor
        return x.numpy()
    return np.asarray(x)


def load_dataset_splits(path):
    """Load a dataset via load_data_medical and return numpy arrays + index splits."""
    resolved = resolve_dataset_path(path)
    _, features, labels, idx_train, idx_val, idx_test = load_data_medical(resolved, train_ratio=0.6, test_ratio=0.2)

    X = to_numpy(features).astype(np.float64)
    y = to_numpy(labels).astype(int).ravel()
    idx_train = to_numpy(idx_train).astype(int).ravel()
    idx_val = to_numpy(idx_val).astype(int).ravel()
    idx_test = to_numpy(idx_test).astype(int).ravel()
    return X, y, idx_train, idx_val, idx_test


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def get_model_factories():
    """Return constructors so a fresh, unfitted model is created for each run."""
    return {
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "ANN (MLP)": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score, is_binary):
    """Compute accuracy, macro F1, binary F1, and ROC-AUC for one model run."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if is_binary:
        f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_score[:, 1])
        except ValueError:
            auc = np.nan
    else:
        f1_binary = np.nan
        try:
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except ValueError:
            auc = np.nan

    return {
        "Accuracy": acc,
        "Macro F1": f1_macro,
        "Binary F1": f1_binary,
        "ROC-AUC": auc,
    }


# --------------------------------------------------------------------------- #
# Training / evaluation for one condition (raw or SMOTE)
# --------------------------------------------------------------------------- #
def run_condition(dataset_name, condition, X_train, y_train, X_test, y_test, is_binary):
    """Train all 5 models under one condition and return metrics + fitted models."""
    results = []
    fitted_models = {}

    for model_name, factory in get_model_factories().items():
        model = factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_score, is_binary)
        metrics.update({"Dataset": dataset_name, "Condition": condition, "Model": model_name})
        results.append(metrics)
        fitted_models[model_name] = model

    return results, fitted_models


def apply_smote(X_train, y_train):
    """Oversample the minority class(es) in the training split only."""
    counts = np.bincount(y_train)
    minority_count = counts[counts > 0].min()
    k_neighbors = max(1, min(5, minority_count - 1))

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# --------------------------------------------------------------------------- #
# Prediction export (for main.py's consolidated ROC / confusion-matrix images)
# --------------------------------------------------------------------------- #
def save_traditional_predictions(dataset_name, fitted_models, X_test, y_test, output_dir):
    """
    Save each SMOTE-trained model's test-set predictions to disk.

    main.py later loads this file together with RA-GCN's own predictions
    pickle to build one combined ROC plot and one combined confusion-matrix
    grid per dataset, covering all 6 models (RA-GCN + 5 traditional) side
    by side, instead of separate per-model figures.
    """
    predictions = {}
    for model_name, model in fitted_models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        predictions[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_score": y_score,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_traditional_predictions.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"[Predictions Saved Successfully] -> Finished exporting {out_path}")


# --------------------------------------------------------------------------- #
# Per-dataset pipeline
# --------------------------------------------------------------------------- #
def benchmark_dataset(path, output_dir, timestamp):
    dataset_name = os.path.splitext(os.path.basename(path))[0]
    
    X, y, idx_train, idx_val, idx_test = load_dataset_splits(path)
    is_binary = len(np.unique(y)) == 2

    # Match the exact training split configuration used by RA-GCN.
    # Features here already come out of load_data_medical() PCA-transformed
    # and scaled once — no additional scaling here, so traditional models
    # see the identical feature representation RA-GCN sees.
    X_train_raw, y_train_raw = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    all_results = []

    # 1) Raw imbalanced training
    raw_results, _ = run_condition(
        dataset_name, "Raw", X_train_raw, y_train_raw, X_test, y_test, is_binary
    )
    all_results.extend(raw_results)

    # 2) SMOTE-rebalanced training
    X_train_smote, y_train_smote = apply_smote(X_train_raw, y_train_raw)
    smote_results, smote_models = run_condition(
        dataset_name, "SMOTE", X_train_smote, y_train_smote, X_test, y_test, is_binary
    )
    all_results.extend(smote_results)

    # Export SMOTE-condition predictions for the consolidated cross-model
    # ROC / confusion-matrix images built later in main.py
    if is_binary:
        save_traditional_predictions(dataset_name, smote_models, X_test, y_test, output_dir)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file.")
    # Redirect default path out of root into a dedicated folder
    parser.add_argument('--output-dir', default="results_dashboard", help="Directory to save artifacts.")
    args = parser.parse_args()

    # Generate isolated clean script timestamp
    runtime_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_results = benchmark_dataset(args.dataset, args.output_dir, runtime_timestamp)
    
    # Save individual dataframe output as a backup csv within target directory
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_out_path = os.path.join(args.output_dir, f"{dataset_name}_traditional_metrics_{runtime_timestamp}.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"[Metrics Saved Successfully] -> Finished exporting {csv_out_path}")


if __name__ == "__main__":
    main()
