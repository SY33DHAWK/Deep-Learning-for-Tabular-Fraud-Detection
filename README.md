# 🛡️ Deep Learning for Tabular Fraud Detection

A production-grade, end-to-end Deep Learning pipeline for predicting fraudulent transactions on the IEEE-CIS Fraud Detection dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Weights & Biases](https://img.shields.io/badge/Tracking-Weights_%26_Biases-ffbe00.svg)

##  Problem Statement

Financial fraud detection is a classic, highly imbalanced classification problem (only **3.5% fraud**). While tree-based models dominate this space, this project demonstrates that Deep Learning can achieve competitive results through proper handling of high-cardinality categorical features and rigorous MLOps practices.

##  Key Results

| Metric | Score |
|--------|-------|
| **Validation ROC-AUC** | **0.7989** |
| **Fraud Recall** | 53% (at 0.5 threshold) |
| **Model Size** | 458K parameters |
| **Training Time** | ~15 min on CPU |

## 🏗️ Architecture Highlights

### 1. **Data Engineering**
- **Memory-efficient preprocessing** using Polars (95% size reduction with Parquet)
- **Categorical embeddings** for high-cardinality features (e.g., `card1` with 13,554 unique values)
- **Robust imputation**: Median for numerical, "UNKNOWN" token for categorical

### 2. **Model Architecture (Tabular MLP)**