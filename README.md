# 🛡️ Deep Learning for Tabular Fraud Detection

A production-grade, end-to-end Deep Learning pipeline for predicting fraudulent transactions on the IEEE-CIS Fraud Detection dataset. This project demonstrates advanced tabular data handling, custom PyTorch architectures, and rigorous MLOps practices.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Weights & Biases](https://img.shields.io/badge/Tracking-Weights_%26_Biases-ffbe00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Problem Statement

Financial fraud detection is a classic, highly imbalanced classification problem (only **~3.5% fraud rate**). While tree-based ensembles (XGBoost, LightGBM) traditionally dominate this space, this project demonstrates how Deep Learning can achieve competitive results through proper handling of high-cardinality categorical features, robust data engineering, and rigorous MLOps practices.

## 🏆 Key Results

Evaluated on a 10% held-out validation set (59,054 samples).

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Validation ROC-AUC** | **0.7989** | Threshold-independent measure of model quality |
| **Fraud Recall** | 53.0% | At default 0.5 threshold |
| **Model Size** | ~458K parameters | Lightweight and fast inference |
| **Training Time** | ~15 mins | On standard CPU (faster with GPU + Mixed Precision) |

---

## 🏗️ Architecture & Engineering Decisions

### 1. Data Engineering (Polars + PyTorch)
* **Memory Efficiency:** Raw CSVs are preprocessed using `polars` and saved as `.parquet`, reducing disk footprint by ~95% and enabling lightning-fast I/O.
* **Categorical Embeddings:** Instead of memory-intensive One-Hot Encoding for high-cardinality features (e.g., `card1` with 13,554 unique values), the pipeline uses dedicated `nn.Embedding` layers to learn dense, meaningful representations.
* **Robust Imputation:** Numerical features are imputed with column medians; categorical features are mapped to a dedicated `"UNKNOWN"` token (index 0) to handle missing data gracefully.

### 2. Model Architecture (Tabular MLP)
```text
Input: 401 Numerical Features + 20 Categorical Features
       ↓
Categorical Embeddings (dim=16 per feature)
       ↓
Concatenation (Total Input Dim: 401 + (20 * 16) = 721)
       ↓
Linear(721 → 256) + BatchNorm1d + ReLU + Dropout(0.3)
       ↓
Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
       ↓
Linear(128 → 64)  + BatchNorm1d + ReLU + Dropout(0.3)
       ↓
Output: 1 Logit (Binary Classification via BCEWithLogitsLoss)
