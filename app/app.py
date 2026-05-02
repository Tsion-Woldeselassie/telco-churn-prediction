import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #F8FAFC; }
.metric-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid #0D7377; margin-bottom: 12px;
}
.risk-high  { background: #FEF2F2; border-left-color: #DC2626; }
.risk-low   { background: #F0FDF4; border-left-color: #16A34A; }
.risk-medium{ background: #FFFBEB; border-left-color: #F59E0B; }
.insight-box {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 14px; color: #1E3A5F;
}
.warning-box {
    background: #FFF7ED; border: 1px solid #FED7AA;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 14px; color: #92400E;
}
.info-row { display: flex; justify-content: space-between; font-size: 13px; color: #475569; margin-bottom: 4px; }
div[data-testid="stSidebar"] { background: #0A1628; }
div[data-testid="stSidebar"] label { color: #D0DCE8 !important; }
div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: white; }
div[data-testid="stSidebar"] p { color: #B0C4D8; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    d = joblib.load("churn_model_deployment.pkl")
    explainer = shap.TreeExplainer(d["model"])
    return d, explainer

deployment, explainer = load_model()
THRESHOLD = deployment["optimal_threshold"]
FEATURE_COLS = deployment["feature_cols"]

FEATURE_DISPLAY = {
    "gender": "Gender (Male)", "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner", "Dependents": "Has Dependents",
    "tenure": "Tenure (months)", "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines", "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup", "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies", "PaperlessBilling": "Paperless Billing",
    "MonthlyCharges": "Monthly Charges ($)", "TotalCharges": "Total Charges ($)",
    "InternetService_DSL": "Internet: DSL",
    "InternetService_Fiber optic": "Internet: Fiber Optic",
    "InternetService_No": "Internet: None",
    "Contract_Month-to-month": "Contract: Month-to-Month",
    "Contract_One year": "Contract: One Year",
    "Contract_Two year": "Contract: Two Year",
    "PaymentMethod_Bank transfer (automatic)": "Payment: Bank Transfer",
    "PaymentMethod_Credit card (automatic)": "Payment: Credit Card",
    "PaymentMethod_Electronic check": "Payment: Electronic Check",
    "PaymentMethod_Mailed check": "Payment: Mailed Check",
}

with st.sidebar:
    st.markdown("## 📡 ChurnGuard")
    st.markdown("*Customer Retention Intelligence*")
    st.markdown("---")
    st.markdown("**Contract & Service**")
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    internet = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "None"])
    payment  = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check",
                                                "Bank Transfer (Auto)", "Credit Card (Auto)"])
    st.markdown("**Account Info**")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charge = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges  = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                      value=float(tenure * monthly_charge), step=10.0)
    st.markdown("**Demographics**")
    senior     = st.checkbox("Senior Citizen (65+)")
    partner    = st.checkbox("Has Partner")
    dependents = st.checkbox("Has Dependents")
    gender_m   = st.checkbox("Male")
    st.markdown("**Services**")
    col1, col2 = st.columns(2)
    with col1:
        phone    = st.checkbox("Phone", value=True)
        multi    = st.checkbox("Multi-Lines")
        security = st.checkbox("Security")
        backup   = st.checkbox("Backup")
    with col2:
        device   = st.checkbox("Device Prot.")
        tech     = st.checkbox("Tech Support")
        tv       = st.checkbox("Streaming TV")
        movies   = st.checkbox("Streaming Movies")
    paperless = st.checkbox("Paperless Billing", value=True)
    st.markdown("---")
    predict_btn = st.button("🔍  Predict Churn Risk", use_container_width=True, type="primary")

def build_features():
    row = {col: 0 for col in FEATURE_COLS}
    row["gender"]          = 1 if gender_m else 0
    row["SeniorCitizen"]   = 1 if senior else 0
    row["Partner"]         = 1 if partner else 0
    row["Dependents"]      = 1 if dependents else 0
    row["tenure"]          = tenure
    row["PhoneService"]    = 1 if phone else 0
    row["MultipleLines"]   = 1 if multi else 0
    row["OnlineSecurity"]  = 1 if security else 0
    row["OnlineBackup"]    = 1 if backup else 0
    row["DeviceProtection"]= 1 if device else 0
    row["TechSupport"]     = 1 if tech else 0
    row["StreamingTV"]     = 1 if tv else 0
    row["StreamingMovies"] = 1 if movies else 0
    row["PaperlessBilling"]= 1 if paperless else 0
    row["MonthlyCharges"]  = monthly_charge
    row["TotalCharges"]    = total_charges
    row["InternetService_Fiber optic"] = 1 if internet == "Fiber Optic" else 0
    row["InternetService_DSL"]         = 1 if internet == "DSL" else 0
    row["InternetService_No"]          = 1 if internet == "None" else 0
    row["Contract_Month-to-month"]     = 1 if contract == "Month-to-Month" else 0
    row["Contract_One year"]           = 1 if contract == "One Year" else 0
    row["Contract_Two year"]           = 1 if contract == "Two Year" else 0
    row["PaymentMethod_Electronic check"]         = 1 if payment == "Electronic Check" else 0
    row["PaymentMethod_Mailed check"]             = 1 if payment == "Mailed Check" else 0
    row["PaymentMethod_Bank transfer (automatic)"]= 1 if payment == "Bank Transfer (Auto)" else 0
    row["PaymentMethod_Credit card (automatic)"]  = 1 if payment == "Credit Card (Auto)" else 0
    return pd.DataFrame([row])

def make_shap_chart(shap_vals, feat_names, n=8):
    pairs   = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:n]
    names   = [FEATURE_DISPLAY.get(p[0], p[0]) for p in pairs][::-1]
    vals    = [p[1] for p in pairs][::-1]
    colors  = ["#DC2626" if v > 0 else "#2563EB" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.barh(names, vals, color=colors, height=0.55, edgecolor="none")
    ax.axvline(0, color="#94A3B8", linewidth=0.8)
    ax.set_xlabel("SHAP contribution  (→ churn,  ← stay)", fontsize=9, color="#64748B")
    ax.tick_params(axis="y", labelsize=9); ax.tick_params(axis="x", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color("#E2E8F0")
    ax.legend(handles=[mpatches.Patch(color="#DC2626", label="↑ churn risk"),
                        mpatches.Patch(color="#2563EB", label="↓ churn risk")],
              fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout(pad=0.5)
    return fig

def make_gauge(prob):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    for start, end, color in [(0,0.33,"#DCFCE7"),(0.33,0.66,"#FEF9C3"),(0.66,1.0,"#FEE2E2")]:
        theta = np.linspace(np.pi*(1-end), np.pi*(1-start), 100)
        ax.fill_between(0.5+0.42*np.cos(theta), 0.15, 0.15+0.42*np.sin(theta),
                        color=color, alpha=0.8)
    angle = np.pi*(1-prob)
    needle_color = "#DC2626" if prob >= THRESHOLD else "#16A34A" if prob < 0.33 else "#F59E0B"
    ax.annotate("", xy=(0.5+0.38*np.cos(angle), 0.15+0.38*np.sin(angle)), xytext=(0.5,0.15),
                arrowprops=dict(arrowstyle="-|>", color=needle_color, lw=2.5, mutation_scale=14))
    ax.plot(0.5, 0.15, "o", color=needle_color, markersize=7, zorder=5)
    ax.text(0.08,0.18,"LOW",  fontsize=8, color="#16A34A", fontweight="bold")
    ax.text(0.46,0.58,"MED",  fontsize=8, color="#D97706", fontweight="bold")
    ax.text(0.84,0.18,"HIGH", fontsize=8, color="#DC2626", fontweight="bold")
    ax.text(0.5,-0.08,f"{prob*100:.1f}%", fontsize=20, fontweight="bold",
            ha="center", va="center", color=needle_color, transform=ax.transAxes)
    plt.tight_layout(pad=0)
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Telco Churn Risk Predictor")
st.markdown(
    "Fill in the customer profile on the left, then click **Predict Churn Risk**. "
    "The model returns a churn probability, a CHURN or STAY decision, and the top factors "
    "driving that prediction."
)
st.markdown("---")

if not predict_btn:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### How this works")
        st.markdown(f"""
This tool uses a machine learning model trained on 7,032 real telecom customers
to predict which customers are at risk of cancelling their service.

**Model:** XGBoost + SMOTEENN  
**Performance:** ROC-AUC = 0.82 · Recall = 0.79 · F1 = 0.62  
**Decision threshold:** {THRESHOLD:.2f} — if the probability is at or above this, the customer is flagged as CHURN risk.
        """)
        st.markdown("### Limitations & warnings")
        st.markdown("""
- The model was trained on IBM Telco data (US market). Results may vary for other markets.
- The model catches ~4 out of 5 actual churners, but also flags some customers who would have stayed.
- This tool supports retention decisions — it does not replace human judgment.
- Always review the risk factors alongside the score before taking action.
        """)
    with col_b:
        st.markdown("### Top churn signals (from training data)")
        for label, detail in [
            ("🔴 Month-to-month contract", "Churn rate: 42% vs 3% for 2-year contracts"),
            ("🔴 Fiber optic internet",    "Churn rate: 41% — likely price sensitivity"),
            ("🔴 Tenure under 12 months",  "First year is the highest-risk period"),
            ("🟢 Two-year contract",        "Churn rate: only 3%"),
            ("🟢 Online Security + Tech Support", "Add-ons reduce churn by ~2×"),
            ("🟢 Tenure over 48 months",   "Long-tenured customers rarely leave"),
        ]:
            st.markdown(f"**{label}**  \n{detail}")

if predict_btn:
    features_df = build_features()
    scaled      = deployment["scaler"].transform(features_df)
    scaled_df   = pd.DataFrame(scaled, columns=FEATURE_COLS)
    prob        = deployment["model"].predict_proba(scaled)[0][1]
    decision    = "CHURN" if prob >= THRESHOLD else "STAY"
    shap_vals   = explainer.shap_values(scaled_df)[0]

    if prob >= THRESHOLD:
        risk_level = "HIGH RISK"; risk_class = "risk-high"
    elif prob >= 0.33:
        risk_level = "MODERATE RISK"; risk_class = "risk-medium"
    else:
        risk_level = "LOW RISK"; risk_class = "risk-low"

    col1, col2, col3 = st.columns([1.2, 1, 1.8])
    with col1:
        st.markdown("#### Churn Probability")
        st.pyplot(make_gauge(prob), use_container_width=True); plt.close()
    with col2:
        st.markdown("#### Decision")
        st.markdown(f"""
<div class="metric-card {risk_class}" style="margin-top:8px;">
  <div style="font-size:13px;color:#64748B;margin-bottom:6px;">Model output</div>
  <div style="font-size:28px;font-weight:700;color:{'#DC2626' if decision=='CHURN' else '#16A34A'}">{decision}</div>
  <div style="font-size:13px;color:#475569;margin-top:4px;">{risk_level}</div>
  <div style="font-size:12px;color:#94A3B8;margin-top:8px;">Threshold: {THRESHOLD:.2f} | Score: {prob:.3f}</div>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("#### What this means")
        feat_row = features_df.iloc[0]
        if decision == "CHURN":
            st.markdown(f"""<div class="warning-box">
<strong>This customer is at elevated risk of leaving.</strong><br><br>
Score: <strong>{prob*100:.1f}%</strong> — above the {THRESHOLD*100:.0f}% threshold.
Proactive outreach is recommended before their next billing cycle.
</div>""", unsafe_allow_html=True)
            st.markdown("**Suggested actions:**")
            actions = []
            if feat_row.get("Contract_Month-to-month", 0): actions.append("Offer a discounted annual contract upgrade")
            if feat_row.get("InternetService_Fiber optic", 0): actions.append("Review fiber pricing — consider loyalty rate")
            if not feat_row.get("OnlineSecurity", 0): actions.append("Bundle free Online Security for 3 months")
            if not feat_row.get("TechSupport", 0): actions.append("Offer Tech Support package trial")
            if feat_row.get("tenure", 12) < 12: actions.append("Assign to first-year retention specialist")
            if not actions: actions.append("Contact customer — review their service satisfaction")
            for a in actions[:4]: st.markdown(f"- {a}")
        else:
            st.markdown(f"""<div class="insight-box">
<strong>This customer is likely to stay.</strong><br><br>
Score: <strong>{prob*100:.1f}%</strong> — below the {THRESHOLD*100:.0f}% threshold.
No urgent intervention needed.
</div>""", unsafe_allow_html=True)
            st.markdown("**Recommended:**\n- Continue standard engagement\n- Monitor at next billing cycle\n- Consider loyalty rewards if tenure > 24 months")

    st.markdown("---")
    col4, col5 = st.columns([1.4, 1])
    with col4:
        st.markdown("#### Why this prediction? — Top risk factors")
        st.pyplot(make_shap_chart(shap_vals, FEATURE_COLS, n=8), use_container_width=True); plt.close()
        st.caption("Red bars push toward CHURN. Blue bars push toward STAY. Powered by SHAP.")
    with col5:
        st.markdown("#### Customer profile summary")
        for label, val in [
            ("Contract", contract), ("Internet", internet), ("Payment", payment),
            ("Tenure", f"{tenure} months"), ("Monthly", f"${monthly_charge}"),
            ("Total paid", f"${total_charges:,.0f}"), ("Senior", "Yes" if senior else "No"),
            ("Security", "Yes" if security else "No"), ("Tech Support", "Yes" if tech else "No"),
        ]:
            st.markdown(f'<div class="info-row"><span>{label}</span><strong>{val}</strong></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Model information**")
        for label, val in [
            ("Algorithm","XGBoost + SMOTEENN"), ("ROC-AUC","0.8233"),
            ("Recall","79.4% of churners caught"), ("Threshold",f"{THRESHOLD:.2f}"),
        ]:
            st.markdown(f'<div class="info-row"><span>{label}</span><strong>{val}</strong></div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("⚠️  Limitations & responsible use"):
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**What the model can get wrong**")
            st.markdown("- **False positives (51% precision):** About half of flagged customers would not have actually churned.\n- **False negatives:** The model misses ~1 in 5 actual churners.\n- **Data drift:** Retrain every 6–12 months if the customer mix changes significantly.")
        with rc:
            st.markdown("**Responsible use guidelines**")
            st.markdown("- Use this score as one input — not the only input.\n- Do not use this model for pricing or eligibility decisions.\n- Retention offers should be personalised — look at the risk factors, not just the score.")

st.markdown("---")
st.markdown("<div style='text-align:center;font-size:12px;color:#94A3B8;'>ChurnGuard · Tsion Woldeselassie · Capstone 2 · XGBoost + SMOTEENN · AUC 0.82</div>", unsafe_allow_html=True)
