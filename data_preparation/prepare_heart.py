"""
data_preparation/prepare_heart.py

Downloads the UCI Heart Disease dataset and converts it into the
(adjacency, features, labels) pickle format expected by code/utils.py.

Graph construction: cosine similarity between standardized patient feature
vectors, with an edge drawn when similarity >= 0.70.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.70
SAVE_PATH = "data/heart/heart_data.pkl"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"


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
    print("Sourcing and cleaning Heart Disease Dataset...")
    df = pd.read_csv(URL)

    # 'thal' is a categorical string column (e.g. "normal", "fixed", "reversible");
    # convert to numeric codes so it can be used as a graph/model feature.
    if 'thal' in df.columns and df['thal'].dtype == 'object':
        df['thal'] = pd.Categorical(df['thal']).codes

    feature_cols = [c for c in df.columns if c != 'target']
    build_and_save_graph(df, feature_cols, 'target', SAVE_PATH, SIMILARITY_THRESHOLD)


if __name__ == "__main__":
    main()
