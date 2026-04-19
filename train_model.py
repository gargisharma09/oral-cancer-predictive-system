"""
train_model.py
==============
Trains a balanced Random Forest on the oral cancer dataset and serialises
the fitted pipeline to model.pkl.

Clinical note
-------------
In medical screening, a FALSE NEGATIVE (predicting "no cancer" when the
patient actually has cancer) is far more dangerous than a false positive.

We therefore optimise for RECALL on the positive class, not raw accuracy.
The model uses class_weight='balanced' and the primary reported metric is
recall, not accuracy.

Usage
-----
    python train_model.py

Output
------
  model.pkl       – serialised sklearn pipeline (scaler + RF classifier)
  feature_names.pkl – ordered list of feature column names used at training
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    accuracy_score,
)

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

print("=" * 60)
print("Oral Cancer Prediction — Model Training")
print("=" * 60)
print(f"\nLoading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Drop leakage / ID columns ─────────────────────────────────────────────
# These columns are downstream of the diagnosis target and must be excluded.
LEAKAGE_COLS = [
    "ID",
    "Country",
    "Cancer Stage",
    "Treatment Type",
    "Survival Rate (5-Year, %)",
    "Cost of Treatment (USD)",
    "Economic Burden (Lost Workdays per Year)",
]
TARGET_COL = "Oral Cancer (Diagnosis)"

existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
df.drop(columns=existing_leakage, inplace=True)
print(f"Dropped leakage/ID columns: {existing_leakage}")

# ── 3. Encode categoricals ────────────────────────────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ── 4. Split features / target ────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

FEATURE_NAMES = list(X.columns)
print(f"\nFeatures used ({len(FEATURE_NAMES)}):")
for fn in FEATURE_NAMES:
    print(f"  - {fn}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")
print(f"Positive-class prevalence — Train: {y_train.mean():.3f}  Test: {y_test.mean():.3f}")

# ── 5. Build pipeline ─────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "clf",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",   # ← key: maximises recall on minority class
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

# ── 6. Cross-validation (recall focus) ────────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation (scorer = recall) …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_recall = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall")
cv_auc    = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"  CV Recall  : {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"  CV ROC-AUC : {cv_auc.mean():.4f}   ± {cv_auc.std():.4f}")

# ── 7. Fit on full training set ───────────────────────────────────────────────
print("\nFitting final model on full training set …")
pipeline.fit(X_train, y_train)

# ── 8. Evaluate on held-out test set ─────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc    = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)
cm     = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("TEST-SET PERFORMANCE")
print("=" * 60)
print(f"Accuracy  : {acc:.4f}")
print(f"Recall    : {recall:.4f}   <- PRIMARY METRIC (minimise false negatives)")
print(f"ROC-AUC   : {auc:.4f}")
print("\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  !! False Negatives = {cm[1,0]}  (missed cancer cases -- the critical error)")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Cancer", "Cancer"]))

# ── 9. Feature importances ────────────────────────────────────────────────────
importances = pipeline.named_steps["clf"].feature_importances_
feat_imp = (
    pd.Series(importances, index=FEATURE_NAMES)
    .sort_values(ascending=False)
)
print("Top-10 Feature Importances:")
for feat, imp in feat_imp.head(10).items():
    bar = "#" * int(imp * 100)
    print(f"  {feat:<45s} {imp:.4f}  {bar}")

# ── 10. Serialise ─────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(FEATURE_NAMES, FEATURES_PATH)
print(f"\n[OK] Model saved    -> {MODEL_PATH}")
print(f"[OK] Features saved -> {FEATURES_PATH}")
print("\nTraining complete.")
