"""
app.py – OralGuard AI  |  Oral Cancer Risk Predictor (Streamlit Demo)
=====================================================================
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
    page_title="OralGuard AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
    <style>
    /* ---- Hide Streamlit chrome ---- */
    #MainMenu, header, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ---- Global ---- */
    html, body, .stApp {
        font-family: 'DM Sans', sans-serif;
        background-color: #F0F4F8;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0C4A6E 0%, #0E7490 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        font-weight: 500;
        padding: 6px 0;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.25) !important;
    }

    /* ---- Cards ---- */
    .card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border-left: 5px solid #0E7490;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border-left: 5px solid #0E7490;
        text-align: center;
    }
    .stat-card .stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #0C4A6E;
        margin: 0;
    }
    .stat-card .stat-label {
        font-size: 0.9rem;
        color: #5B7A8C;
        margin-top: 0.3rem;
    }

    /* ---- Result cards ---- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 5px solid #DC2626;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 4px 20px rgba(220,38,38,0.12);
        animation: fadeIn 0.45s ease;
        margin-bottom: 1rem;
    }
    .result-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left: 5px solid #16A34A;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 4px 20px rgba(22,163,74,0.12);
        animation: fadeIn 0.45s ease;
        margin-bottom: 1rem;
    }
    .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        margin: 0 0 0.4rem 0;
    }
    .result-sub {
        font-size: 0.95rem;
        opacity: 0.8;
        margin: 0;
    }

    /* ---- Predict button ---- */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #0C4A6E, #0E7490);
        color: #FFFFFF;
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        box-shadow: 0 4px 14px rgba(14,116,144,0.3);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14,116,144,0.45);
    }

    /* ---- Section labels ---- */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #0E7490;
        margin: 1.2rem 0 0.4rem 0;
    }
    .disclaimer-banner {
        background: #FFF8E7;
        border-left: 4px solid #D97706;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #78350F;
        margin-bottom: 1rem;
    }

    /* ---- Headings ---- */
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #0C4A6E; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model artefacts (auto-train if missing) ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

@st.cache_resource
def load_model():
    """Load the trained pipeline.  If pkl files are absent, train on-the-fly."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        import subprocess, sys
        train_script = os.path.join(BASE_DIR, "train_model.py")
        with st.spinner("⏳ First-run setup: training model (takes ~60 s)…"):
            subprocess.run([sys.executable, train_script], check=True)
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names

try:
    pipeline, FEATURE_NAMES = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:DM Serif Display,serif;font-size:1.6rem;margin-bottom:0;'>🦷 OralGuard AI</h2>"
        "<p style='font-size:0.8rem;opacity:0.75;margin-top:4px;'>AI-Powered Oral Cancer Screening</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Predict", "📊 Data Insights", "📈 Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;opacity:0.65;line-height:1.5;'>"
        "⚠️ <strong>Disclaimer:</strong> This tool is for educational and portfolio demonstration "
        "purposes only. It is <em>not</em> a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider."
        "</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        "<h1 style='font-size:3rem;line-height:1.15;margin-bottom:0.5rem;'>"
        "Early Detection.<br>Better Outcomes."
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:1.1rem;color:#374151;max-width:640px;line-height:1.7;'>"
        "OralGuard AI is an <strong>AI-powered oral cancer screening assistant</strong> that "
        "analyses over 10 clinical risk factors — including lifestyle habits, medical history, "
        "and oral health indicators — to provide an instant, evidence-informed risk assessment. "
        "Early detection increases 5-year survival rates by up to <strong>80%</strong>."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='stat-card'>"
            "<p class='stat-num'>300K+</p>"
            "<p class='stat-label'>New oral cancer cases diagnosed globally each year</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='stat-card'>"
            "<p class='stat-num'>80%</p>"
            "<p class='stat-label'>5-year survival rate when detected early</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='stat-card'>"
            "<p class='stat-num'>10+</p>"
            "<p class='stat-label'>Risk factors analysed by the predictive model</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📖  Why Recall Matters — Our Clinical Philosophy", expanded=False):
        st.markdown(
            """
            In clinical screening, **the cost of errors is not symmetric**.

            | Error type | Meaning | Consequence |
            |---|---|---|
            | **False Positive** | Model says *cancer*, patient is healthy | Extra tests, anxiety — manageable |
            | **False Negative** | Model says *healthy*, patient has cancer | Missed diagnosis — **potentially fatal** |

            > A false negative is far more dangerous in a clinical setting.

            The Random Forest is trained with `class_weight='balanced'`. The primary metric is **recall** (sensitivity), ensuring every true positive is flagged even at the cost of some false alarms.
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown("<h2>Patient Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='disclaimer-banner'>"
        "<strong>Educational Tool Only</strong> — This application is for portfolio demonstration "
        "and educational purposes. It is <em>not</em> a substitute for professional medical advice."
        "</div>",
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.error("Model file not found. Please run `python train_model.py` first.")
        st.stop()

    col_l, col_r = st.columns(2)

    with col_l:
        # 👤 Demographics
        st.markdown("<p class='section-label'>👤 Demographics</p>", unsafe_allow_html=True)
        age = st.slider("Age", min_value=18, max_value=90, value=45, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_val = 0 if gender == "Female" else 1

        # 🚬 Lifestyle
        st.markdown("<p class='section-label'>🚬 Lifestyle</p>", unsafe_allow_html=True)
        tobacco = st.selectbox("Tobacco Use", ["No", "Yes"])
        tobacco_val = 1 if tobacco == "Yes" else 0

        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        alcohol_val = 1 if alcohol == "Yes" else 0

        betel = st.selectbox("Betel Quid Use", ["No", "Yes"])
        betel_val = 1 if betel == "Yes" else 0

        sun = st.selectbox("Chronic Sun Exposure", ["No", "Yes"])
        sun_val = 1 if sun == "Yes" else 0

        diet = st.selectbox("Diet (Fruits & Vegetables Intake)", ["High", "Moderate", "Low"])
        diet_map = {"High": 0, "Moderate": 2, "Low": 1}
        diet_val = diet_map[diet]

    with col_r:
        # 🏥 Medical History
        st.markdown("<p class='section-label'>🏥 Medical History</p>", unsafe_allow_html=True)
        hpv = st.selectbox("HPV Infection", ["No", "Yes"])
        hpv_val = 1 if hpv == "Yes" else 0

        family_hist = st.selectbox("Family History of Cancer", ["No", "Yes"])
        family_val = 1 if family_hist == "Yes" else 0

        immune = st.selectbox("Compromised Immune System", ["No", "Yes"])
        immune_val = 1 if immune == "Yes" else 0

        early_dx = st.selectbox("Early Diagnosis", ["No", "Yes"])
        early_val = 1 if early_dx == "Yes" else 0

        # 🦷 Oral Health
        st.markdown("<p class='section-label'>🦷 Oral Health</p>", unsafe_allow_html=True)
        hygiene = st.selectbox("Poor Oral Hygiene", ["No", "Yes"])
        hygiene_val = 1 if hygiene == "Yes" else 0

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

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Cancer Risk", use_container_width=True)

    # ── Build input DataFrame (UNCHANGED prediction logic) ──────────────────
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
        with st.spinner("Analyzing patient risk factors..."):
            prediction    = pipeline.predict(input_data)[0]
            prob_cancer   = pipeline.predict_proba(input_data)[0][1]
            prob_pct      = prob_cancer * 100

        # Risk tier
        if prob_pct >= 70:
            risk_label = "HIGH RISK"
            risk_emoji = "🔴"
            bar_color  = "#DC2626"
        elif prob_pct >= 40:
            risk_label = "MODERATE RISK"
            risk_emoji = "🟡"
            bar_color  = "#D97706"
        else:
            risk_label = "LOW RISK"
            risk_emoji = "🟢"
            bar_color  = "#16A34A"

        diagnosis = "Oral Cancer Detected" if prediction == 1 else "No Cancer Detected"
        is_malignant = prediction == 1

        # Styled result card
        card_class = "result-high" if is_malignant else "result-low"
        title_color = "#DC2626" if is_malignant else "#16A34A"
        st.markdown(
            f"<div class='{card_class}'>"
            f"<p class='result-title' style='color:{title_color};'>{risk_emoji} {risk_label} — {diagnosis}</p>"
            f"<p class='result-sub'>Estimated cancer probability: <strong>{prob_pct:.1f}%</strong></p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Confidence progress bar
        st.markdown(f"**Confidence score:** `{prob_pct:.1f}%`")
        st.progress(min(int(prob_pct), 100))

        # Probability gauge (matplotlib)
        st.markdown("#### Probability Gauge")
        fig_gauge, ax_gauge = plt.subplots(figsize=(7, 1.2))
        fig_gauge.patch.set_facecolor("#F0F4F8")
        ax_gauge.set_facecolor("#F0F4F8")
        ax_gauge.barh(0, 100, height=0.5, color="#E2E8F0", left=0)
        ax_gauge.barh(0, prob_pct, height=0.5, color=bar_color, left=0, alpha=0.85)
        ax_gauge.axvline(40, color="#D97706", linewidth=1.2, linestyle="--", alpha=0.6)
        ax_gauge.axvline(70, color="#DC2626", linewidth=1.2, linestyle="--", alpha=0.6)
        ax_gauge.set_xlim(0, 100)
        ax_gauge.set_ylim(-0.6, 0.6)
        ax_gauge.set_xticks([0, 20, 40, 60, 70, 80, 100])
        ax_gauge.set_xticklabels(["0%","20%","40%","60%","70%","80%","100%"], color="#374151", fontsize=8)
        ax_gauge.set_yticks([])
        for spine in ax_gauge.spines.values():
            spine.set_visible(False)
        ax_gauge.text(prob_pct + 1, 0, f"{prob_pct:.1f}%", va="center", color="#0C4A6E", fontsize=9, fontweight="bold")
        fig_gauge.tight_layout()
        st.pyplot(fig_gauge)
        plt.close(fig_gauge)

        # Disclaimer warning below result
        st.warning(
            "⚠️ **Medical Disclaimer:** This result is generated by an AI model trained on synthetic data "
            "and is intended for educational purposes only. It must **not** be used for clinical decision-making. "
            "Please consult a qualified healthcare professional for any medical concerns."
        )

        st.markdown("---")

    # Feature importance (always visible on predict page)
    st.markdown("### Top-10 Feature Importances")
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_series = pd.Series(importances, index=FEATURE_NAMES).sort_values(ascending=True).tail(10)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    fig_imp.patch.set_facecolor("#F0F4F8")
    ax_imp.set_facecolor("#F0F4F8")
    colors = ["#0E7490" if v < 0.5 else "#DC2626" for v in feat_series.values]
    ax_imp.barh(feat_series.index, feat_series.values, color=colors, edgecolor="none", height=0.6)
    ax_imp.set_xlabel("Importance", color="#374151")
    ax_imp.tick_params(colors="#374151", labelsize=9)
    for spine in ax_imp.spines.values():
        spine.set_color("#CBD5E1")
    legend_patches = [
        mpatches.Patch(color="#DC2626", label="Dominant feature"),
        mpatches.Patch(color="#0E7490", label="Supporting feature"),
    ]
    ax_imp.legend(handles=legend_patches, framealpha=0, labelcolor="#374151", fontsize=8)
    fig_imp.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.markdown("<h2>Data Insights</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card'>"
        "<strong>Dataset:</strong> 84,922 synthetic patient records with 25 clinical features "
        "(sourced from Kaggle — Oral Cancer Prediction dataset)."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About the Dataset & Leakage Guard", expanded=True):
        st.markdown(
            """
            **Model**: Random Forest Classifier (200 trees, `class_weight='balanced'`, StandardScaler preprocessing), trained on **17 non-leakage features**.

            **Leakage guard**: The model intentionally excludes Cancer Stage, Survival Rate, Treatment Type, and Cost of Treatment — these are downstream of the diagnosis target.

            **Key Features Used**:
            | Feature | Type |
            |---|---|
            | Age | Numeric |
            | Gender | Binary |
            | Tobacco Use | Binary |
            | Alcohol Consumption | Binary |
            | HPV Infection | Binary |
            | Betel Quid Use | Binary |
            | Chronic Sun Exposure | Binary |
            | Poor Oral Hygiene | Binary |
            | Diet (Fruits & Veg) | Categorical |
            | Family History | Binary |
            | Immune Status | Binary |
            | Oral Lesions | Binary |
            | Unexplained Bleeding | Binary |
            | Difficulty Swallowing | Binary |
            | White/Red Patches | Binary |
            | Tumor Size (cm) | Numeric |
            | Early Diagnosis | Binary |
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("<h2>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card'>"
        "Metrics computed on a <strong>held-out 20% test set</strong>. "
        "The primary metric is <strong>Recall</strong> to minimise false negatives in clinical screening."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall (Cancer)", "1.00")
    col2.metric("Precision (Cancer)", "1.00")
    col3.metric("ROC-AUC", "1.00")
    col4.metric("Accuracy", "1.00")

    st.info(
        "**Note on perfect scores:** These reflect the structure of the synthetic dataset "
        "(Tumor Size is a near-perfect separator). Real-world clinical models would require "
        "substantially more nuanced feature engineering and larger real patient cohorts."
    )

    with st.expander("📖 Why Recall Over Accuracy?", expanded=False):
        st.markdown(
            """
            | Error type | Meaning | Consequence |
            |---|---|---|
            | **False Positive** | Model says *cancer*, patient healthy | Extra tests, anxiety — manageable |
            | **False Negative** | Model says *healthy*, patient has cancer | Missed diagnosis — **potentially fatal** |

            For this reason the model uses `class_weight='balanced'` and prioritises **Recall (Sensitivity)**.
            """
        )

    st.markdown("---")
    st.markdown(
        "**Source code:** [github.com/gargisharma09](https://github.com/gargisharma09)"
    )
