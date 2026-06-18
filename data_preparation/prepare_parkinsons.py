"""
data_preparation/prepare_parkinsons.py

Downloads the UCI Parkinson's voice dataset and converts it into the
(adjacency, features, labels) pickle format expected by code/utils.py.

Graph construction: cosine similarity between standardized patient feature
vectors, with an edge drawn when similarity >= 0.85.

Note: with 22 raw features, this dataset triggers the automatic PCA
reduction step in code/utils.py (features > 10 are reduced to 5 principal
components before graph use / training).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.85
SAVE_PATH = "data/parkinsons/parkinsons_data.pkl"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


def build_and_save_graph(df, feature_cols, label_col, save_path, similarity_threshold):
    features = df[feature_cols].values
    labels = df[label_col].values.astype(int)

    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    sim_matrix = cosine_similarity(scaled_features)
    adj = (sim_matrix >= similarity_threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    with open(save_path, "wb") as f:
        pickle.dump((adj, features, labels), f)

    print(f"Successfully processed: {save_path}")
    print(f" -> Patients (Nodes): {adj.shape[0]} | Clinical Features: {features.shape[1]}")
    print(f" -> Class Balance: {np.bincount(labels)} | Generated Edges: {int(adj.sum() / 2)}\n")


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Sourcing Parkinson's Dataset...")
    df = pd.read_csv(URL)
    # 'name' is a non-numeric subject identifier; 'status' is the label
    # (1 = Parkinson's, 0 = healthy).
    feature_cols = [c for c in df.columns if c not in ['name', 'status']]
    build_and_save_graph(df, feature_cols, 'status', SAVE_PATH, SIMILARITY_THRESHOLD)


if __name__ == "__main__":
    main()
