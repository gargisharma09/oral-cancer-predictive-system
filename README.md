# Oral Cancer Risk Predictor

A machine learning project built to predict oral cancer risk using clinical data. Included is a trained Random Forest model and a Streamlit web application for interactive predictions.

## Overview

Oral cancer is often diagnosed in later stages when survival rates are significantly lower. This project aims to demonstrate a predictive risk model using 17 clinical risk factors (such as age, tobacco/alcohol use, oral lesions) to flag high-risk cases early.

**Note on metrics:** In a medical screening scenario, false negatives (missing a cancer case) are typically more dangerous than false positives. Because of this, the primary metric optimized during model training is **Recall** rather than just overall accuracy. 

## Dataset

The dataset used is from Kaggle and contains roughly 85,000 synthetic patient records. 
Features occurring *after* diagnosis (like Cancer Stage or Treatment Type) were intentionally held out to avoid data leakage.

Key predictive features include:
* Tumor Size 
* Presence of Oral Lesions
* White/Red Patches (Leukoplakia)
* Unexplained Bleeding
* Tobacco & Alcohol use

## Setup and Installation

### Running locally

1. Install requirements in a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Generate the model artifacts:
```bash
python train_model.py
```
*(This will generate `model.pkl` and `feature_names.pkl`)*

3. Run the Streamlit app:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501`.

### Running with Docker

You can also run the application isolated inside a Docker container. 

```bash
# Build the image
docker build -t oral-cancer-predictor .

# Run the container
docker run -p 8501:8501 oral-cancer-predictor
```

## Model Performance

The current pipeline uses a standard `StandardScaler` followed by a `RandomForestClassifier` (with balanced class weights). 
Evaluated on a 20% holdout set, the baseline metrics for the Random Forest are:

* **Accuracy:** ~1.00
* **Recall:** ~1.00
* **ROC-AUC:** ~1.00

*(Note: The perfect precision and recall scores here are largely a byproduct of the synthetic nature of this specific Kaggle dataset, particularly concerning the Tumor Size variable. In a real clinical setting, baseline scores would be lower).*

## Next Steps / Future Work

* **Model Explainability:** Add SHAP values (waterfall/beeswarm plots) to understand what specific features led to a given prediction.
* **Hyperparameter Tuning:** Implement GridSearch/RandomSearchCV across the Random Forest parameters instead of using the base settings.
* **Real-world Validation:** Test the pipeline against a real-world, non-synthetic clinical dataset (if available).

## Disclaimer
This project is meant for educational and portfolio demonstration only, using synthetic data. It is not professional medical software.
