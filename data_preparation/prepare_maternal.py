"""
data_preparation/prepare_maternal.py

Downloads the Maternal Health Risk dataset and converts it into the
(adjacency, features, labels) pickle format expected by code/utils.py.

Graph construction: cosine similarity between standardized patient feature
vectors, with an edge drawn when similarity >= 0.85.

Label: binarized from the original 3-level RiskLevel column
("low risk" / "mid risk" / "high risk") into high risk (1) vs. not high
risk (0).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.85
SAVE_PATH = "data/maternal/maternal_data.pkl"
URL = "https://raw.githubusercontent.com/LinkedInLearning/data-centric-ai-4516168/main/Maternal%20Health%20Risk%20Data%20Set.csv"


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Sourcing Maternal Health Risk Dataset...")

    df = pd.read_csv(URL)
    df.columns = [c.strip().lower() for c in df.columns]

    # Binarize: "high risk" -> 1, everything else ("low risk", "mid risk") -> 0
    df['target'] = df['risklevel'].apply(lambda x: 1 if 'high' in str(x).lower() else 0)

    feature_cols = [c for c in df.columns if c not in ['risklevel', 'target']]
    features = df[feature_cols].values
    labels = df['target'].values.astype(int)

    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    sim_matrix = cosine_similarity(scaled_features)
    adj = (sim_matrix >= SIMILARITY_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 0)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump((adj, features, labels), f)

    print(f"Successfully processed: {SAVE_PATH}")
    print(f" -> Patients (Nodes): {adj.shape[0]} | Clinical Features: {features.shape[1]}")
    print(f" -> Columns parsed: {feature_cols}")
    print(f" -> Class Balance: {np.bincount(labels)} | Generated Edges: {int(adj.sum() / 2)}\n")


if __name__ == "__main__":
    main()
