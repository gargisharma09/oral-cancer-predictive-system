# 🦷 OralGuard AI — Oral Cancer Predictive System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

--

## Overview

**OralGuard AI** is an AI-powered oral cancer risk screening system. It analyses patient clinical data — including lifestyle habits, medical history, and oral health indicators — and delivers an instant risk assessment powered by a Random Forest classifier.

Oral cancer claims thousands of lives annually, yet the **5-year survival rate jumps to ~80% when detected early**. OralGuard AI makes screening knowledge accessible and explainable, and serves as a portfolio demonstration of end-to-end machine learning applied to a real public-health problem.

---

## Features

| Feature | Description |
|---|---|
| 🔮 **Risk Prediction** | Binary classification — Cancer Detected / Not Detected |
| 📊 **Confidence Score** | Probability score with visual progress bar |
| 📉 **Feature Importance** | Top-10 feature importances chart |
| 🧭 **EDA Dashboard** | Data Insights page with feature descriptions |
| 📈 **Model Metrics** | Recall, Precision, ROC-AUC, Accuracy at a glance |
| 🐳 **Docker Ready** | Single command to build & run the containerised app |

---

## ML Pipeline

```
Raw Data (CSV)
    │
    ▼
Preprocessing
 (Drop leakage cols, label-encode categoricals)
    │
    ▼
Feature Engineering
 (17 non-leakage clinical features selected)
    │
    ▼
Training
 (Random Forest · 200 trees · class_weight='balanced' · StandardScaler)
    │
    ▼
Prediction
 (model.pkl + feature_names.pkl → Streamlit UI)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML | scikit-learn (Random Forest, StandardScaler) |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Frontend | Streamlit |
| Serialisation | joblib |
| Containerisation | Docker |
| Notebook | Jupyter |

---

## Project Structure

```
oral-cancer-predictive-system/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── dataset.csv             # Source dataset (Kaggle)
├── model.pkl               # Trained pipeline (generated)
├── feature_names.pkl       # Feature name list (generated)
├── Prediction.ipynb        # EDA & modelling notebook
└── finalproject.ipynb      # Extended analysis notebook
```

---

## Dataset

Synthetic dataset of **84,922 patient records** with 25 clinical features (Kaggle — Oral Cancer Prediction).

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age (18–90) |
| Gender | Binary | 0 = Female, 1 = Male |
| Tobacco Use | Binary | Smoker / non-smoker |
| Alcohol Consumption | Binary | Drinker / non-drinker |
| HPV Status | Binary | HPV-positive / negative |
| Betel Nut Use | Binary | Betel quid chewer |
| Family History | Binary | Family history of cancer |
| Previous Lesions | Binary | Prior oral lesion history |
| Dental Visits | Binary | Regular dental check-ups |
| Oral Hygiene | Binary | Poor oral hygiene indicator |

> **Leakage guard**: Cancer Stage, Survival Rate, Treatment Type, and Cost of Treatment are excluded — these are downstream of the diagnosis target.

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/gargisharma09/oral-cancer-predictive-system.git
cd oral-cancer-predictive-system

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates model.pkl and feature_names.pkl)
python train_model.py

# 5. Launch the Streamlit app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker Instructions

```bash
# Build the image
docker build -t oralguard-ai .

# Run the container
docker run -p 8501:8501 oralguard-ai
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Performance

Metrics computed on a **held-out 20% test set**:

| Metric | Score |
|---|---|
| Recall (Cancer class) | XX% |
| Precision (Cancer class) | XX% |
| ROC-AUC | XX% |
| Accuracy | XX% |

> *Note: The model prioritises **Recall** to minimise false negatives — a missed cancer diagnosis is far more dangerous than a false alarm.*

---

## Roadmap

- [x] Data cleaning & EDA
- [x] Random Forest classifier with balanced class weights
- [x] Streamlit live demo
- [x] Docker containerisation
- [x] Feature importance visualisation
- [ ] SHAP explainability plots
- [ ] Real patient dataset integration
- [ ] REST API endpoint (FastAPI)
- [ ] Automated CI/CD pipeline

---

## ⚠️ Medical Disclaimer

This application is for **educational and portfolio demonstration purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. The model is trained on synthetic data and must not be used for clinical decision-making. Always consult a qualified healthcare professional for any medical concerns.

---

## Author

**Gargi Sharma** · [@gargisharma09](https://github.com/gargisharma09)
