# ECG Heartbeat Classification (PyTorch)

Classifies ECG heartbeats from MIT-BIH using classic ML baselines and a 1D-CNN. Includes preprocessing, EDA, imbalance handling, training (≤25 epochs), evaluation (precision/recall/F1, ROC/PR), and SQL experiment logging.

## Tech
Python, PyTorch, scikit-learn, Pandas, Matplotlib, SQLite

## What’s inside
- EDA: class distribution, example beats, imbalance fixes
- Baselines: Logistic Regression, Random Forest
- Deep Learning: 1D-CNN (PyTorch), class-weighted loss
- Metrics: precision/recall/F1, ROC-AUC, PR-AUC; loss/accuracy curves
- Repro: run configs + SQLite logging

## Quick start
1. `pip install -r requirements.txt`
2. Place `mitbih_train.csv`, `mitbih_test.csv` in `data/`
3. Run `notebooks/ecg_cnn.ipynb`

## Results (example — replace with yours)
| Model              | F1 (macro) | ROC-AUC | Notes          |
|-------------------|------------:|--------:|----------------|
| LogisticRegression | 0.84       | 0.93    | Baseline       |
| RandomForest       | 0.86       | 0.95    | Baseline       |
| **1D-CNN**         | **0.90**   | **0.97**| Best overall   |

## Report & Screens
- Loss/Accuracy curves
- ROC/PR curves per class
- Confusion matrix
- Class distribution and sample plots
