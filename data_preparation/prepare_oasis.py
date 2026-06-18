"""
data_preparation/prepare_oasis.py

Converts the OASIS-2 longitudinal dataset CSV into the (adjacency, features,
labels) pickle format expected by code/utils.py.

Graph construction: patients are connected if their MMSE (Mini-Mental State
Examination) scores are equal (gamma = 0, i.e. exact match on this integer
score). This produces a denser graph (density ~0.18, avg degree ~67) than
the other four datasets in this benchmark. An earlier, sparser age-based
graph construction was also tested but is not the version used for the
reported RA-GCN results.

Label: binarized from CDR (Clinical Dementia Rating).
  CDR = 0   -> 0 (Non-demented)
  CDR > 0   -> 1 (Demented)
"""

import os
import pandas as pd
import numpy as np
import pickle

CSV_PATH = "data/oasis/oasis_longitudinal.csv"
OUT_PATH = "data/oasis/oasis_data.pkl"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load data
    df = pd.read_csv(CSV_PATH)

    # Remove rows with missing MMSE
    df = df.dropna(subset=["MMSE"])

    # Fill missing SES values with median
    df["SES"] = df["SES"].fillna(df["SES"].median())

    # Binary label from CDR
    # 0 = Nondemented, >0 = Demented
    labels = (df["CDR"] > 0).astype(int).values

    # Node features
    df["M_F"] = (df["M/F"] == "M").astype(int)

    features = df[
        ["Age", "EDUC", "SES", "M_F", "eTIV", "nWBV", "ASF"]
    ].values.astype(np.float32)

    # Standardize features (z-score)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Graph construction feature: MMSE (cognitive test score)
    mmse = df["MMSE"].values

    # Graph threshold gamma = 0 (exact MMSE score match)
    gamma = 0.0

    adj = (np.abs(mmse[:, None] - mmse[None, :]) <= gamma).astype(np.float32)

    # Remove self loops (utils.py adds them back at load time)
    np.fill_diagonal(adj, 0)

    with open(OUT_PATH, "wb") as f:
        pickle.dump((adj, features, labels), f)

    print("saved:", OUT_PATH)
    print("adj shape:", adj.shape)
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    print("class counts:", np.bincount(labels))
    print("graph density:", adj.sum() / (adj.shape[0] ** 2))


if __name__ == "__main__":
    main()
