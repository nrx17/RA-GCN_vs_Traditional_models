# Evaluation and Imbalance Stress-Testing of RA-GCN

This repository contains an independent extension and benchmarking study built around **RA-GCN** (Re-weighted Adversarial Graph Convolutional Network for disease prediction problems with imbalanced data).

## Credits & References

The core model architecture and training logic are derived from the original implementation released by the paper's authors:

- **Core Paper:** Ghorbani, M., Kazi, A., Baghshah, M. S., Rabiee, H. R., & Navab, N. (2022). RA-GCN: Graph convolutional network for disease prediction problems with imbalanced data. *Medical Image Analysis*, 75, 102272. https://doi.org/10.1016/j.media.2021.102272
- **Preprint:** https://arxiv.org/abs/2103.00221
- **Official code:** https://github.com/mahsa91/RA-GCN-MedIA2022
- **Underlying GCN Architecture:** Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*. https://arxiv.org/abs/1609.02907

```bibtex
@article{ghorbani2022ra,
  title={Ra-gcn: Graph convolutional network for disease prediction problems with imbalanced data},
  author={Ghorbani, Mahsa and Kazi, Anees and Baghshah, Mahdieh Soleymani and Rabiee, Hamid R and Navab, Nassir},
  journal={Medical Image Analysis},
  volume={75},
  pages={102272},
  year={2022},
  publisher={Elsevier}
}
```

---

## What This Repository Does

The original paper evaluated RA-GCN's ability to handle class imbalance using synthetic data and three real clinical datasets (Pima Diabetes, PPMI, Haberman). This repository extends that evaluation in a different direction: RA-GCN is benchmarked against **five real-world clinical datasets** and **five traditional machine learning classifiers** (Decision Tree, Random Forest, KNN, Logistic Regression, and an ANN), looking at how performance changes with dataset size, feature dimensionality, and the use of SMOTE-based resampling.

Five independent medical tasks are covered: OASIS (Alzheimer's/dementia), Diabetes, Parkinson's, Heart Disease, and Maternal Health Risk. Both RA-GCN and the five traditional baselines are evaluated on the exact same patient-level train/validation/test split for each dataset (via a shared data loader), so differences in results reflect the modeling approach rather than differences in how the data was split. An evaluation pipeline automatically produces a 3-panel dashboard (run summary, confusion matrix, ROC curve) for RA-GCN, and a separate ROC comparison plot and metrics CSV for the traditional baselines, for every dataset.

---

## Datasets

| Dataset | Domain | Nodes | Raw Features | Graph Construction | Edge Count | Class Distribution (0 / 1) |
|---|---|---|---|---|---|---|
| OASIS Longitudinal | Alzheimer's / dementia | 371 | 7 (+1 used for graph) | Distance threshold on MMSE (γ=1, i.e. exact score match) | 12,454 | 206 / 165 |
| Diabetes (Pima Indian) | Diabetes risk | 768 | 8 | Cosine similarity ≥ 0.80 | 5,796 | 500 / 268 |
| Parkinson's (voice) | Parkinson's disease | 195 | 22 | Cosine similarity ≥ 0.85 | 460 | 48 / 147 |
| Heart Disease (UCI) | Cardiovascular disease | 303 | 13 | Cosine similarity ≥ 0.70 | 760 | 220 / 83 |
| Maternal Health Risk | Pregnancy risk | 1,014 | 6 | Cosine similarity ≥ 0.85 | 27,003 | 742 / 272 |

All node/edge counts and class distributions above were checked directly against the prepared `.pkl` files. OASIS connects patients whose MMSE score (a cognitive test score) is an exact match; the other four datasets connect patients using cosine similarity between standardized feature vectors, each with its own threshold. Self-loops are excluded from the saved adjacency matrices; `code/utils.py` adds them back, along with row-normalization, when the data is loaded.

Data sources: OASIS Longitudinal ([MRI and Alzheimer's, OASIS-2 longitudinal data, via Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data?select=oasis_longitudinal.csv)), Diabetes ([Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)), Heart Disease ([UCI Heart Disease, via TensorFlow datasets mirror](https://storage.googleapis.com/download.tensorflow.org/data/heart.csv)), Parkinson's ([UCI Parkinson's voice dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)), Maternal Health Risk ([Maternal Health Risk Data Set](https://raw.githubusercontent.com/LinkedInLearning/data-centric-ai-4516168/main/Maternal%20Health%20Risk%20Data%20Set.csv)). Scripts to download and convert each dataset into the `.pkl` format expected by `utils.py` are in `data_preparation/`.

OASIS node features: Age, Education, Socioeconomic Status, Sex, eTIV, normalized whole-brain volume (nWBV), ASF (MMSE is used only for graph construction and is excluded from the node features). The label is binarized from CDR (CDR = 0 → Non-demented; CDR > 0 → Demented). Note: the OASIS-2 longitudinal data includes multiple visits for the same subjects (150 unique subjects across 371 visit-rows), which is a known limitation worth accounting for in any downstream analysis. The resulting graph is denser (density ≈ 0.18, average degree ≈ 67) than the other four datasets in this study; a sparser age-based graph was also tried earlier but is not the version used for the results reported here.

## Using SMOTE on the Traditional Baselines

Each of the five traditional models was trained twice per dataset: once on the raw imbalanced training split, and once on a SMOTE-rebalanced version of the same training split (SMOTE is applied only to the training data, never to validation or test). RA-GCN itself was not trained with SMOTE — generating synthetic graph nodes is not straightforward, since a synthetic patient has no well-defined relationship to the rest of the population graph, and the original paper makes the same point when explaining why resampling is difficult for graph-structured data.

Full metrics for every model, dataset, and condition (raw / SMOTE) are saved in `results_dashboard/*_traditional_metrics*.csv`. Best-performing traditional model per dataset and condition, by Macro F1:

| Dataset | Best Raw Model (Macro F1) | Best SMOTE Model (Macro F1) |
|---|---|---|
| Diabetes | Random Forest (0.704) | Logistic Regression (0.731) |
| Heart Disease | Random Forest (0.723) | Logistic Regression (0.823) |
| Maternal Health | Random Forest (0.922) | Random Forest (0.926) |
| OASIS | Random Forest (0.863) | Random Forest (0.837) |
| Parkinson's | KNN (0.965) | Decision Tree (0.967) |

ROC curves for the SMOTE-trained traditional models, per dataset:

![Diabetes Traditional Models ROC (SMOTE)](results_dashboard/diabetes_data_traditional_roc_smote.png)
![Heart Disease Traditional Models ROC (SMOTE)](results_dashboard/heart_data_traditional_roc_smote.png)
![Maternal Health Traditional Models ROC (SMOTE)](results_dashboard/maternal_data_traditional_roc_smote.png)
![OASIS Traditional Models ROC (SMOTE)](results_dashboard/oasis_data_traditional_roc_smote.png)
![Parkinson's Traditional Models ROC (SMOTE)](results_dashboard/parkinsons_data_traditional_roc_smote.png)

Full metrics tables (Raw and SMOTE, all five models) per dataset:

- `results_dashboard/diabetes_data_traditional_metrics.csv`
- `results_dashboard/heart_data_traditional_metrics.csv`
- `results_dashboard/maternal_data_traditional_metrics.csv`
- `results_dashboard/oasis_data_traditional_metrics.csv`
- `results_dashboard/parkinsons_data_traditional_metrics.csv`

It was this SMOTE comparison that first surfaced a problem with RA-GCN itself: on Parkinson's and Heart Disease, RA-GCN's confusion matrix showed every test patient being assigned to a single class, with Binary F1 = 0.0000 and ROC AUC = 0.5000 (equivalent to random guessing). This did not happen on the lower-dimensional datasets (Diabetes, Maternal, OASIS). The next section covers this collapse and how it was addressed.

## RA-GCN: Before and After PCA

**Before PCA.** RA-GCN's classifier collapsed to predicting a single class for every test node on the two datasets with more than 10 raw features — Parkinson's (22 features) and Heart Disease (13 features). Both show the same failure pattern: Binary F1 = 0.0000, ROC AUC = 0.5000, and a confusion matrix with every prediction landing in one column.

![Diabetes RA-GCN Dashboard, before PCA](results_dashboard/diabetes_data_evaluation_dashboard.png)
![Heart Disease RA-GCN Dashboard, before PCA](results_dashboard/heart_data_evaluation_dashboard.png)
![Maternal Health RA-GCN Dashboard, before PCA](results_dashboard/maternal_data_evaluation_dashboard.png)
![OASIS RA-GCN Dashboard, before PCA](results_dashboard/oasis_data_evaluation_dashboard.png)
![Parkinson's RA-GCN Dashboard, before PCA](results_dashboard/parkinsons_data_evaluation_dashboard.png)

**After PCA.** To address this, `code/utils.py` was changed to automatically standardize and reduce any dataset with more than 10 raw features down to 5 principal components before graph construction and training. Datasets with 10 or fewer raw features (Diabetes, Maternal, OASIS) are left unmodified. Variance retained by PCA: 88.30% for Parkinson's, 60.27% for Heart Disease.

![Diabetes RA-GCN Dashboard, after PCA](results_dashboard/diabetes_data_evaluation_dashboard_20260617_085651.png)
![Heart Disease RA-GCN Dashboard, after PCA](results_dashboard/heart_data_evaluation_dashboard_20260617_085838.png)
![Maternal Health RA-GCN Dashboard, after PCA](results_dashboard/maternal_data_evaluation_dashboard_20260617_085948.png)
![OASIS RA-GCN Dashboard, after PCA](results_dashboard/oasis_data_evaluation_dashboard_20260617_085548.png)
![Parkinson's RA-GCN Dashboard, after PCA](results_dashboard/parkinsons_data_evaluation_dashboard_20260617_085745.png)

With PCA in place, both datasets moved from random-guessing behavior to usable classifiers (Heart Disease: Binary F1 0.7027, AUC 0.9158; Parkinson's: Binary F1 0.7143, AUC 0.9517 — see the full results table below). Why GCN-based weighting networks are particularly prone to this kind of collapse on higher-dimensional, lower-sample data is not fully explained here and would be worth further investigation. It is consistent, though, with a similar observation in the original RA-GCN paper, where its highest-dimensional real dataset (PPMI, 300 features) was reported as the most difficult of the three it tested.

Files in `results_dashboard/` without a timestamp in the filename were generated before this PCA change; files with a timestamp were generated after it.

## RA-GCN Results

RA-GCN was trained for 1000 epochs on each dataset using the original adversarial training procedure (a classifier plus class-specific weighting networks), with no resampling applied. Results below are from the best validation-selected epoch per dataset, using the PCA-reduced feature set for Parkinson's and Heart Disease.

| Dataset | Test Nodes | Accuracy | Macro F1 | Binary F1 | ROC AUC |
|---|---|---|---|---|---|
| Diabetes | 154 | 0.7013 | 0.6747 | 0.5818 | 0.7524 |
| Heart Disease | 61 | 0.8197 | 0.7866 | 0.7027 | 0.9158 |
| Maternal Health | 203 | 0.8621 | 0.8274 | 0.7500 | 0.8493 |
| OASIS | 75 | 0.8800 | 0.8768 | 0.8571 | 0.9188 |
| Parkinson's | 39 | 0.7949 | 0.7771 | 0.7143 | 0.9517 |

Confusion matrices (rows = true class, columns = predicted class):

```
Diabetes:     [[76, 24], [22, 32]]
Heart:        [[37,  7], [ 4, 13]]
Maternal:     [[133,16], [12, 42]]
OASIS:        [[39,  3], [ 6, 27]]
Parkinson's:  [[10,  0], [ 8, 21]]
```

Full dashboards (run summary, confusion matrix, ROC curve) for each dataset's final, PCA-corrected result:

![Diabetes RA-GCN Dashboard](results_dashboard/diabetes_data_evaluation_dashboard_20260617_085651.png)
![Heart Disease RA-GCN Dashboard](results_dashboard/heart_data_evaluation_dashboard_20260617_085838.png)
![Maternal Health RA-GCN Dashboard](results_dashboard/maternal_data_evaluation_dashboard_20260617_085948.png)
![OASIS RA-GCN Dashboard](results_dashboard/oasis_data_evaluation_dashboard_20260617_085548.png)
![Parkinson's RA-GCN Dashboard](results_dashboard/parkinsons_data_evaluation_dashboard_20260617_085745.png)

## Summary of Findings

RA-GCN does not come out ahead of the traditional baselines on every dataset here. On Diabetes, Heart Disease, and Parkinson's, the strongest traditional model — often Random Forest, or a SMOTE-boosted Logistic Regression or Decision Tree — reaches a higher Macro F1 than RA-GCN. RA-GCN does best on OASIS, where it beats most of the traditional baselines on Binary F1, the metric most sensitive to catching the minority (Demented) class.

Taken together, this suggests RA-GCN's adversarial reweighting is not an automatic improvement over simpler models paired with standard imbalance-handling techniques, and that its advantage, where it shows up, may depend on dataset size, feature dimensionality, and the structure of the population graph.

## Repository Structure

```
data/                     Prepared (adjacency, features, labels) pickles per dataset
data_preparation/         Scripts to download raw data and build each dataset's .pkl file
code/
  layer.py                Graph convolution layer
  model.py                RA-GCN classifier + weighting networks
  main_medical.py         RA-GCN training entry point, generates evaluation dashboards
  traditional_baselines.py Decision Tree / Random Forest / KNN / Logistic Regression / ANN benchmark, raw + SMOTE
  utils.py                Shared data loader, PCA preprocessing, metrics
main.py                   Runs both pipelines across all 5 datasets
generate_dataset.py       Builds a synthetic dataset (data/synthetic/per-90gt-0.5.pkl); not used in the reported benchmark
results_dashboard/        Saved metrics CSVs and dashboard images
```

## Limitations

- Each traditional model and RA-GCN configuration was run on a single fixed train/val/test split rather than averaged over multiple random seeds; the numbers above are single-run results, not means ± standard deviation as reported in the original paper.
- The OASIS dataset includes repeated subjects across visit-rows, which may inflate performance if the same subject appears in both train and test.
- RA-GCN without PCA collapses entirely on the two higher-dimensional datasets used here (Heart Disease, Parkinson's); this sensitivity is not yet fully understood.
- SMOTE's `k_neighbors` parameter is adapted per dataset based on minority class size rather than fixed, since some datasets have too few minority samples for a fixed neighbor count.
- A synthetic dataset (`data/synthetic/per-90gt-0.5.pkl`, built by `generate_dataset.py`) is included in this repository but was not used in the reported benchmark.

## Note on AI Tool Usage

AI tools were used to assist with coding, debugging, and documentation. All results were independently verified before inclusion in this report.
