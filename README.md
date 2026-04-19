# 🦷 Oral Cancer Risk Predictor

> **A machine-learning pipeline that predicts oral cancer diagnosis from clinical patient data — with a live interactive Streamlit demo.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## Table of Contents
1. [Project Goal](#-project-goal)
2. [Why Recall — Not Accuracy — Is the Right Metric](#-why-recall-not-accuracy-is-the-right-metric)
3. [Dataset](#-dataset)
4. [Features Used](#-features-used)
5. [Model Architecture](#-model-architecture)
6. [Performance](#-performance)
7. [Run Locally](#-run-locally)
8. [Run with Docker](#-run-with-docker)
9. [Project Structure](#-project-structure)

---

## 🎯 Project Goal

Build a machine-learning model that predicts whether a patient has oral cancer (`Oral Cancer (Diagnosis)`) from clinical risk-factor data, expose it through an interactive web demo, and package the whole stack in Docker — so a recruiter or clinician can interact with the model **without cloning or running a single notebook**.

---

## ⚠️ Why Recall — Not Accuracy — Is the Right Metric

Most student ML projects report accuracy as the headline number.  
In a **medical screening context, that is the wrong metric to optimise.**

| Error type | Plain English | Clinical consequence |
|---|---|---|
| **False Positive** | Model says "cancer" — patient is healthy | Extra tests, some anxiety — manageable |
| **False Negative** | Model says "healthy" — patient has cancer | **Missed diagnosis, delayed treatment — potentially fatal** |

> **A false negative — missing a real cancer case — is far more dangerous than a false positive in a clinical setting.**

This project therefore:
- Trains the Random Forest with **`class_weight='balanced'`** to avoid the majority-class shortcut.
- Reports **recall** as the primary metric in every comparison table.
- Uses **ROC-AUC** as the secondary metric (threshold-free, tells the full story).
- Accuracy is reported for completeness, but it is explicitly *not* the optimisation target.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Oral Cancer Prediction Dataset |
| Records | **84 922** patient rows |
| Total columns | 25 (incl. target) |
| Target column | `Oral Cancer (Diagnosis)` — binary Yes/No |
| Class balance | ~50 / 50 (balanced dataset) |

---

## 🔬 Features Used

The following 17 **pre-diagnosis** risk factors are used as model inputs.  
Downstream columns (`Cancer Stage`, `Treatment Type`, `Survival Rate (5-Year, %)`, etc.) are **excluded** to prevent data leakage.

| # | Feature | Type |
|---|---|---|
| 1 | Age | Numeric |
| 2 | Gender | Binary |
| 3 | Tobacco Use | Binary |
| 4 | Alcohol Consumption | Binary |
| 5 | HPV Infection | Binary |
| 6 | Betel Quid Use | Binary |
| 7 | Chronic Sun Exposure | Binary |
| 8 | Poor Oral Hygiene | Binary |
| 9 | Diet (Fruits & Vegetables Intake) | Ordinal |
| 10 | Family History of Cancer | Binary |
| 11 | Compromised Immune System | Binary |
| 12 | Oral Lesions | Binary |
| 13 | Unexplained Bleeding | Binary |
| 14 | Difficulty Swallowing | Binary |
| 15 | White or Red Patches in Mouth | Binary |
| 16 | Tumor Size (cm) | Numeric |
| 17 | Early Diagnosis | Binary |

---

## 🏗️ Model Architecture

```
Input (17 features)
       │
   StandardScaler          ← zero mean, unit variance
       │
RandomForestClassifier
  n_estimators = 200
  class_weight = "balanced" ← key for recall maximisation
  random_state = 42
       │
  predict / predict_proba
```

**Pipeline** is serialised with `joblib` → `model.pkl`

---

## 📈 Performance

Evaluated on a stratified 20% held-out test set (16 985 records).

| Metric | Score | Note |
|---|---|---|
| **Recall (Cancer)** | **1.00** | ← primary clinical metric |
| Precision (Cancer) | 1.00 | |
| F1-Score (Cancer) | 1.00 | |
| ROC-AUC | 1.00 | |
| Accuracy | 1.00 | |

Cross-validation (5-fold, recall scorer): **1.00 ± 0.00**

> **Note on perfect scores:** These scores reflect the structure of this *synthetic* dataset — `Tumor Size (cm)` is a near-perfect linear separator of the target.  
> In real-world clinical ML, perfect scores should be treated with extreme scepticism and investigated for leakage.  
> **The clinical framing, leakage-prevention decisions, and emphasis on recall are the transferable skills demonstrated here**, not the numbers themselves.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- `pip`

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/gargisharma09/Predictive-System.git
cd Predictive-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model.pkl)
python train_model.py

# 4. Launch the Streamlit demo
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Run with Docker

```bash
# Build the image (train_model.py runs inside; pre-generate model.pkl first)
python train_model.py          # generates model.pkl locally

docker build -t oral-cancer .
docker run -p 8501:8501 oral-cancer
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
.
├── dataset.csv            # 84 922 patient records
├── requirements.txt       # pip dependencies
├── Dockerfile             # Docker build → streamlit run app.py
├── train_model.py         # training script → saves model.pkl
├── model.pkl              # serialised Random Forest pipeline
├── feature_names.pkl      # ordered feature list (used by app.py)
├── app.py                 # Streamlit interactive demo
├── Prediction.ipynb       # exploratory analysis notebook
└── finalproject.ipynb     # full modelling notebook
```

---

*Disclaimer: This project is for educational and portfolio purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*
