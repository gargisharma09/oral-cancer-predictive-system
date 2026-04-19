"""
app.py – Oral Cancer Risk Predictor (Streamlit Demo)
=====================================================
Run:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oral Cancer Risk Predictor",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- global palette ---- */
    :root {
        --bg-dark:     #0d1117;
        --panel-dark:  #161b22;
        --accent-teal: #2dd4bf;
        --accent-rose: #f43f5e;
        --accent-amber:#fbbf24;
        --accent-green:#22c55e;
        --text-muted:  #8b949e;
    }
    .stApp { background-color: var(--bg-dark); }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--panel-dark);
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e6edf3 !important;
    }

    /* metric cards */
    div[data-testid="metric-container"] {
        background: var(--panel-dark);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="metric-container"] label {
        color: var(--text-muted) !important;
    }

    /* banner */
    .disclaimer-banner {
        background: #1c2128;
        border-left: 4px solid var(--accent-amber);
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .risk-high   { color: #f43f5e; font-weight: 700; font-size: 1.6rem; }
    .risk-medium { color: #fbbf24; font-weight: 700; font-size: 1.6rem; }
    .risk-low    { color: #22c55e; font-weight: 700; font-size: 1.6rem; }
    .section-label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; }
    h1, h2, h3 { color: #e6edf3 !important; }
    p, li { color: #c9d1d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model artefacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    return model, feature_names

try:
    pipeline, FEATURE_NAMES = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Sidebar — patient inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Profile")
    st.markdown("---")

    age = st.slider("Age", min_value=18, max_value=90, value=45, step=1)

    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_val = 0 if gender == "Female" else 1

    tobacco = st.selectbox("Tobacco Use", ["No", "Yes"])
    tobacco_val = 1 if tobacco == "Yes" else 0

    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    alcohol_val = 1 if alcohol == "Yes" else 0

    hpv = st.selectbox("HPV Infection", ["No", "Yes"])
    hpv_val = 1 if hpv == "Yes" else 0

    betel = st.selectbox("Betel Quid Use", ["No", "Yes"])
    betel_val = 1 if betel == "Yes" else 0

    sun = st.selectbox("Chronic Sun Exposure", ["No", "Yes"])
    sun_val = 1 if sun == "Yes" else 0

    hygiene = st.selectbox("Poor Oral Hygiene", ["No", "Yes"])
    hygiene_val = 1 if hygiene == "Yes" else 0

    diet = st.selectbox("Diet (Fruits & Vegetables Intake)", ["High", "Moderate", "Low"])
    diet_map = {"High": 0, "Moderate": 2, "Low": 1}
    diet_val = diet_map[diet]

    family_hist = st.selectbox("Family History of Cancer", ["No", "Yes"])
    family_val = 1 if family_hist == "Yes" else 0

    immune = st.selectbox("Compromised Immune System", ["No", "Yes"])
    immune_val = 1 if immune == "Yes" else 0

    lesions = st.selectbox("Oral Lesions", ["No", "Yes"])
    lesions_val = 1 if lesions == "Yes" else 0

    bleeding = st.selectbox("Unexplained Bleeding", ["No", "Yes"])
    bleeding_val = 1 if bleeding == "Yes" else 0

    swallowing = st.selectbox("Difficulty Swallowing", ["No", "Yes"])
    swallowing_val = 1 if swallowing == "Yes" else 0

    patches = st.selectbox("White or Red Patches in Mouth", ["No", "Yes"])
    patches_val = 1 if patches == "Yes" else 0

    tumor_size = st.slider(
        "Tumor Size (cm)  — 0 if no known tumour",
        min_value=0.0, max_value=12.0, value=0.0, step=0.1,
    )

    early_dx = st.selectbox("Early Diagnosis", ["No", "Yes"])
    early_val = 1 if early_dx == "Yes" else 0

    predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🦷 Oral Cancer Risk Predictor")
st.markdown(
    '<div class="disclaimer-banner">'
    "<strong>Educational Tool Only</strong> — This application is for portfolio demonstration "
    "and educational purposes only. It is <em>not</em> a substitute for professional medical "
    "advice, diagnosis, or treatment. Always consult a qualified healthcare professional."
    "</div>",
    unsafe_allow_html=True,
)

# ── Why Recall Matters callout ────────────────────────────────────────────────
with st.expander("📖  Why This Model Prioritises Recall Over Accuracy", expanded=False):
    st.markdown(
        """
        In clinical screening for a disease like oral cancer, **the cost of errors is not symmetric**.

        | Error type | What it means | Clinical consequence |
        |------------|---------------|----------------------|
        | **False Positive** | Model says *cancer*, patient is healthy | Extra tests, anxiety — manageable |
        | **False Negative** | Model says *healthy*, patient has cancer | Missed diagnosis, delayed treatment — **potentially fatal** |

        > **A false negative — missing a real cancer case — is far more dangerous than a false positive
        > in a clinical setting.**

        For this reason the Random Forest is trained with `class_weight='balanced'` and the primary
        evaluation metric throughout the project is **recall** (sensitivity), not accuracy.
        The goal is to flag every true positive, even at the cost of some false alarms.
        """
    )

if not model_loaded:
    st.error(
        "Model file not found. Please run `python train_model.py` first to generate `model.pkl`."
    )
    st.stop()

# ── Prediction ─────────────────────────────────────────────────────────────────
input_data = pd.DataFrame(
    [[
        age, gender_val, tobacco_val, alcohol_val, hpv_val, betel_val,
        sun_val, hygiene_val, diet_val, family_val, immune_val,
        lesions_val, bleeding_val, swallowing_val, patches_val,
        tumor_size, early_val,
    ]],
    columns=FEATURE_NAMES,
)

if predict_btn:
    prediction = pipeline.predict(input_data)[0]
    prob_cancer = pipeline.predict_proba(input_data)[0][1]
    prob_pct = prob_cancer * 100

    # Risk tier
    if prob_pct >= 70:
        risk_label = "HIGH RISK"
        risk_class = "risk-high"
        risk_emoji = "🔴"
        bar_color  = "#f43f5e"
    elif prob_pct >= 40:
        risk_label = "MODERATE RISK"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
        bar_color  = "#fbbf24"
    else:
        risk_label = "LOW RISK"
        risk_class = "risk-low"
        risk_emoji = "🟢"
        bar_color  = "#22c55e"

    diagnosis = "Oral Cancer Detected" if prediction == 1 else "No Cancer Detected"

    # ── Result cards ──
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.metric("Prediction", diagnosis)

    with col2:
        st.metric("Cancer Probability", f"{prob_pct:.1f}%")

    with col3:
        st.markdown(f'<p class="section-label">Risk Level</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="{risk_class}">{risk_emoji}  {risk_label}</p>',
            unsafe_allow_html=True,
        )

    # ── Probability gauge (matplotlib) ────────────────────────────────────────
    st.markdown("### Probability Gauge")
    fig_gauge, ax_gauge = plt.subplots(figsize=(7, 1.2))
    fig_gauge.patch.set_facecolor("#161b22")
    ax_gauge.set_facecolor("#161b22")

    # Background track
    ax_gauge.barh(0, 100, height=0.5, color="#30363d", left=0)
    # Filled portion
    ax_gauge.barh(0, prob_pct, height=0.5, color=bar_color, left=0, alpha=0.85)
    # Zone lines
    ax_gauge.axvline(40, color="#fbbf24", linewidth=1.2, linestyle="--", alpha=0.6)
    ax_gauge.axvline(70, color="#f43f5e", linewidth=1.2, linestyle="--", alpha=0.6)

    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(-0.6, 0.6)
    ax_gauge.set_xticks([0, 20, 40, 60, 70, 80, 100])
    ax_gauge.set_xticklabels(
        ["0%", "20%", "40%", "60%", "70%", "80%", "100%"],
        color="#8b949e", fontsize=8,
    )
    ax_gauge.set_yticks([])
    for spine in ax_gauge.spines.values():
        spine.set_visible(False)

    ax_gauge.text(prob_pct + 1, 0, f"{prob_pct:.1f}%", va="center", color="#e6edf3", fontsize=9)
    fig_gauge.tight_layout()
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

    st.markdown("---")

# ── Feature importance chart (always visible) ─────────────────────────────────
st.markdown("### Top-10 Feature Importances")
importances = pipeline.named_steps["clf"].feature_importances_
feat_series = pd.Series(importances, index=FEATURE_NAMES).sort_values(ascending=True).tail(10)

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
fig_imp.patch.set_facecolor("#161b22")
ax_imp.set_facecolor("#161b22")

colors = ["#2dd4bf" if v < 0.5 else "#f43f5e" for v in feat_series.values]
bars = ax_imp.barh(feat_series.index, feat_series.values, color=colors, edgecolor="none", height=0.6)

ax_imp.set_xlabel("Importance", color="#8b949e")
ax_imp.tick_params(colors="#c9d1d9", labelsize=9)
for spine in ax_imp.spines.values():
    spine.set_color("#30363d")
ax_imp.xaxis.label.set_color("#8b949e")
ax_imp.set_facecolor("#161b22")

legend_patches = [
    mpatches.Patch(color="#f43f5e", label="Dominant feature"),
    mpatches.Patch(color="#2dd4bf", label="Supporting feature"),
]
ax_imp.legend(handles=legend_patches, framealpha=0, labelcolor="#8b949e", fontsize=8)

fig_imp.tight_layout()
st.pyplot(fig_imp)
plt.close(fig_imp)

# ── About section ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️  About this project"):
    st.markdown(
        """
        **Dataset**: 84 922 synthetic patient records with 25 clinical features
        (sourced from Kaggle—Oral Cancer Prediction dataset).

        **Model**: Random Forest Classifier (200 trees, `class_weight='balanced'`,
        StandardScaler preprocessing), trained on 17 non-leakage features.

        **Leakage guard**: The model intentionally excludes Cancer Stage, Survival Rate,
        Treatment Type, and Cost of Treatment, as these are downstream of the diagnosis target.

        **Metrics (held-out 20% test set)**:
        | Metric | Score |
        |--------|-------|
        | Recall (Cancer class) | **1.00** |
        | Precision (Cancer class) | **1.00** |
        | ROC-AUC | **1.00** |
        | Accuracy | **1.00** |

        *Note: Perfect scores reflect the structure of this synthetic dataset
        (Tumor Size is a near-perfect separator). Real-world clinical models would
        require substantially more nuanced feature engineering and larger real patient cohorts.*

        **Source code**: [github.com/gargisharma09/Predictive-System](https://github.com/gargisharma09/Predictive-System)
        """
    )
