import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f5f7ff 0%, #f8fbff 55%, #f1f7ff 100%); }
.block-container { padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1440px; }
.hero { padding: 28px 32px; border-radius: 24px; color: white;
        background: linear-gradient(115deg, #111c44 0%, #253f88 55%, #4668d8 100%);
        box-shadow: 0 14px 35px rgba(29, 48, 104, .18); margin-bottom: 22px; }
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.25rem; font-weight: 800; margin: 0 0 7px 0; }
.hero p { color: #dce5ff; margin: 0; font-size: 1.02rem; }
.eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: .72rem; font-weight: 700; color: #a9c1ff; margin-bottom: 9px; }
.metric-card { background: rgba(255,255,255,.92); border: 1px solid #e4e9f5; border-radius: 18px;
               padding: 18px 20px; box-shadow: 0 5px 18px rgba(30, 50, 100, .045); }
.metric-label { color: #68738c; font-size: .84rem; font-weight: 600; margin-bottom: 6px; }
.metric-value { color: #18264c; font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; }
.metric-note { color: #8993a8; font-size: .76rem; margin-top: 4px; }
.section-title { color: #17264d; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.28rem; margin: 10px 0 5px 0; }
.section-sub { color: #768198; font-size: .9rem; margin-bottom: 14px; }
div[data-testid="stMetric"] { background: white; padding: 15px 18px; border: 1px solid #e4e9f5; border-radius: 16px; }
div[data-testid="stMetricLabel"] { color: #68738c; }
.stButton > button { border-radius: 12px; font-weight: 700; min-height: 2.8rem; }
[data-testid="stSidebar"] { background: #101a3b; }
[data-testid="stSidebar"] * { color: #f1f4ff; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #17264d; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage_log",
    "CustServCalls", "DayMins", "OverageFee_log", "RoamMins",
    "HighServiceCalls", "NoDataEngagement"
]
REQUIRED = ["Churn", "AccountWeeks", "ContractRenewal", "DataPlan",
            "DataUsage", "CustServCalls", "DayMins", "OverageFee", "RoamMins"]

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df["DataUsage_log"] = np.log1p(df["DataUsage"].clip(lower=0))
    df["OverageFee_log"] = np.log1p(df["OverageFee"].clip(lower=0))
    df["HighServiceCalls"] = (df["CustServCalls"] >= 4).astype(int)
    df["NoDataEngagement"] = ((df["DataPlan"] == 0) & (df["DataUsage"] == 0)).astype(int)
    # Ensure target is binary and in expected 0/1 format.
    vals = set(df["Churn"].dropna().unique())
    if not vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError("The Churn column must contain binary values 0 and 1.")
    df["Churn"] = df["Churn"].astype(int)
    if df["Churn"].nunique() < 2:
        raise ValueError("The dataset must contain both Churn classes (0 and 1).")
    return df

@st.cache_data(show_spinner=False)
def train_models(df):
    X = df[FEATURES]
    y = df["Churn"]
    if y.value_counts().min() < 2:
        raise ValueError("Please provide at least two examples from each Churn class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    try:
        from xgboost import XGBClassifier
        neg, pos = np.bincount(y_train.astype(int), minlength=2)
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=max(1, neg / max(pos, 1)), n_estimators=180,
            max_depth=4, learning_rate=0.1, random_state=42,
            eval_metric="logloss", n_jobs=-1
        )
    except Exception:
        pass

    fitted, records, predictions = {}, [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
        fitted[name] = model
        predictions[name] = (pred, prob)
        records.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0
        })
    metrics = pd.DataFrame(records).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return fitted, metrics, predictions, X_test, y_test

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.caption("TELECOM CUSTOMER ANALYTICS")
    st.markdown("---")
    st.markdown("### 1 · Load your data")
    uploaded = st.file_uploader("Upload telecom_churn.csv", type=["csv"])
    st.markdown("### 2 · Navigate")
    page = st.radio("Choose a workspace", ["Overview", "Churn Explorer", "Predict a Customer", "Model Lab"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built with Python · Streamlit · scikit-learn")
    st.caption("Model outputs are estimates, not guarantees.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">CUSTOMER RETENTION ANALYTICS</div>
  <h1>Telecom Churn Intelligence</h1>
  <p>Understand customer behaviour, uncover churn signals, and explore machine-learning predictions.</p>
</div>
""", unsafe_allow_html=True)

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        data = prepare_data(raw)
    except Exception as e:
        st.error(f"Could not load this dataset: {e}")
        st.info("Upload a CSV containing the required telecom churn columns. See the sidebar or the data requirements below.")
        st.stop()
else:
    st.info("👈 Upload your **telecom_churn.csv** in the sidebar to launch the dashboard.")
    with st.expander("Required CSV columns"):
        st.code(", ".join(REQUIRED), language="text")
        st.write("The target column **Churn** must use 0 = stayed and 1 = churned. The app creates the engineered features used by the notebook.")
    st.markdown("### What you can explore")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📊 Churn overview")
        st.write("Review retention rate, customer mix, and high-level dataset quality.")
    with c2:
        st.markdown("#### 🔎 Behaviour signals")
        st.write("Compare service calls, monthly usage, and charges for churned customers.")
    with c3:
        st.markdown("#### 🤖 Model lab")
        st.write("Train and compare classification models on a stratified test split.")
    st.stop()

try:
    models, model_metrics, predictions, X_test, y_test = train_models(data)
except Exception as e:
    st.error(f"Unable to train models: {e}")
    st.stop()

best_model_name = model_metrics.iloc[0]["Model"]
best_model = models[best_model_name]
total = len(data)
churn_count = int(data["Churn"].sum())
churn_rate = data["Churn"].mean() * 100
retained = total - churn_count

if page == "Overview":
    st.markdown('<div class="section-title">Your retention snapshot</div><div class="section-sub">A quick health check of the uploaded customer base.</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    with a: metric_card("Total customers", f"{total:,}", "Rows in the uploaded dataset")
    with b: metric_card("Churn rate", f"{churn_rate:.1f}%", "Customers marked as churned")
    with c: metric_card("Retained", f"{retained:,}", "Customers with Churn = 0")
    with d: metric_card("Top model · ROC-AUC", f"{model_metrics.iloc[0]['ROC-AUC']:.3f}", best_model_name)
    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-title">Customer status mix</div>', unsafe_allow_html=True)
        counts = data["Churn"].map({0: "Retained", 1: "Churned"}).value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(counts, names="Status", values="Customers", hole=0.62,
                     color="Status", color_discrete_map={"Retained": "#526de5", "Churned": "#ff7a8a"})
        fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color="white", width=3)))
        fig.update_layout(margin=dict(t=15,b=15,l=10,r=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="#263457")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Churn by customer service calls</div>', unsafe_allow_html=True)
        rates = data.groupby("CustServCalls", as_index=False)["Churn"].mean()
        rates["Churn rate (%)"] = rates["Churn"] * 100
        fig = px.bar(rates, x="CustServCalls", y="Churn rate (%)", color="Churn rate (%)",
                     color_continuous_scale=["#91a8ff", "#ff7a8a"], labels={"CustServCalls":"Customer service calls"})
        fig.update_layout(margin=dict(t=15,b=15,l=5,r=5), coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#263457")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download cleaned dataset", data.to_csv(index=False).encode("utf-8"), "telecom_churn_cleaned.csv", "text/csv")

elif page == "Churn Explorer":
    st.markdown('<div class="section-title">Explore churn drivers</div><div class="section-sub">Filter the customer base and compare the signals that may be associated with churn.</div>', unsafe_allow_html=True)
    with st.container():
        xcol, ycol = st.columns(2)
        with xcol:
            feature = st.selectbox("Choose a feature", ["CustServCalls", "DayMins", "MonthlyCharge", "OverageFee", "AccountWeeks", "DataUsage", "RoamMins"])
        with ycol:
            segment = st.selectbox("Customer segment", ["All customers", "Retained only", "Churned only"])
    view = data.copy()
    if segment == "Retained only": view = view[view["Churn"] == 0]
    if segment == "Churned only": view = view[view["Churn"] == 1]
    if feature not in view.columns:
        st.warning(f"`{feature}` is not available in this CSV, so showing DataUsage instead.")
        feature = "DataUsage"
    fig = px.histogram(view, x=feature, color=view["Churn"].map({0:"Retained",1:"Churned"}),
                       barmode="overlay", opacity=.72, nbins=35,
                       color_discrete_map={"Retained":"#526de5","Churned":"#ff7a8a"},
                       labels={"color":"Customer status"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#263457")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Average feature value by status")
    summary = data.groupby("Churn")[feature].agg(["mean", "median", "count"]).reset_index()
    summary["Customer status"] = summary["Churn"].map({0:"Retained",1:"Churned"})
    st.dataframe(summary[["Customer status","mean","median","count"]].rename(columns={"mean":"Mean","median":"Median","count":"Customers"}).round(2), use_container_width=True, hide_index=True)
    st.caption("These are descriptive associations in the uploaded data; they do not prove that a feature causes churn.")

elif page == "Predict a Customer":
    st.markdown('<div class="section-title">Individual customer risk check</div><div class="section-sub">Enter a customer profile to estimate churn probability using the highest ROC-AUC model.</div>', unsafe_allow_html=True)
    st.caption(f"Selected model: **{best_model_name}** · Holdout ROC-AUC: **{model_metrics.iloc[0]['ROC-AUC']:.3f}**")
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            weeks = st.number_input("Account age (weeks)", min_value=0, max_value=2000, value=100)
            contract = st.selectbox("Recently renewed contract?", ["Yes", "No"])
            plan = st.selectbox("Has data plan?", ["Yes", "No"])
            usage = st.number_input("Monthly data usage (GB)", min_value=0.0, value=2.0, step=0.5)
        with col2:
            calls = st.number_input("Customer service calls", min_value=0, max_value=100, value=1)
            mins = st.number_input("Daytime minutes / month", min_value=0.0, value=180.0, step=10.0)
            overage = st.number_input("Overage fee", min_value=0.0, value=10.0, step=1.0)
            roam = st.number_input("Roaming minutes", min_value=0.0, value=5.0, step=1.0)
        with col3:
            st.markdown("#### Profile summary")
            st.write("Inputs are transformed to match the feature engineering used in the notebook.")
            st.info("This is a model estimate. Review the model's test metrics before using predictions for decisions.")
        submitted = st.form_submit_button("✨ Estimate churn risk", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "AccountWeeks": weeks, "ContractRenewal": int(contract == "Yes"),
            "DataPlan": int(plan == "Yes"), "DataUsage_log": np.log1p(usage),
            "CustServCalls": calls, "DayMins": mins, "OverageFee_log": np.log1p(overage),
            "RoamMins": roam, "HighServiceCalls": int(calls >= 4),
            "NoDataEngagement": int((plan == "No") and (usage == 0))
        }], columns=FEATURES)
        probability = float(best_model.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        st.markdown("---")
        r1, r2 = st.columns([1, 1.3])
        with r1:
            st.metric("Estimated churn probability", f"{probability:.1%}")
            if prediction:
                st.error("Higher predicted risk · model class: Churn")
            else:
                st.success("Lower predicted risk · model class: Retained")
        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability*100,
                number={"suffix":"%", "font":{"size":34, "color":"#18264c"}},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#ff7a8a" if prediction else "#526de5"},
                       "bgcolor":"white", "borderwidth":0,
                       "steps":[{"range":[0,35],"color":"#e8f7f0"},{"range":[35,65],"color":"#fff4dc"},{"range":[65,100],"color":"#ffe8ec"}]}
            ))
            gauge.update_layout(height=230, margin=dict(t=15,b=10,l=25,r=25), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, use_container_width=True)
        st.caption("The 50% cutoff is a simple default, not a business-validated threshold. Validate calibration and choose a threshold appropriate to your retention strategy.")

elif page == "Model Lab":
    st.markdown('<div class="section-title">Model performance lab</div><div class="section-sub">Models are trained on 80% of the uploaded data and evaluated on a stratified 20% holdout set.</div>', unsafe_allow_html=True)
    st.dataframe(model_metrics.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1 score","ROC-AUC"]}), use_container_width=True, hide_index=True)
    metric = st.selectbox("Compare models by", ["ROC-AUC", "F1 score", "Recall", "Precision", "Accuracy"])
    fig = px.bar(model_metrics.sort_values(metric, ascending=True), x=metric, y="Model", orientation="h",
                 text_auto=".3f", color=metric, color_continuous_scale=["#9fb4ff","#4b63d8"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                      xaxis_range=[0,1], font_color="#263457", margin=dict(t=15,b=15,l=5,r=5))
    st.plotly_chart(fig, use_container_width=True)
    selected = st.selectbox("Inspect confusion matrix", list(models.keys()), index=list(models.keys()).index(best_model_name))
    pred, prob = predictions[selected]
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    fig = px.imshow(cm, text_auto=True, x=["Predicted retained","Predicted churn"],
                    y=["Actual retained","Actual churn"], color_continuous_scale="Blues",
                    labels={"x":"Prediction","y":"Actual","color":"Customers"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#263457")
    st.plotly_chart(fig, use_container_width=True)
    report = classification_report(y_test, pred, target_names=["Retained (0)", "Churned (1)"], output_dict=True, zero_division=0)
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.download_button("⬇ Download model comparison", model_metrics.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")

st.markdown("---")
st.caption("Telecom Churn Intelligence · Use representative, permitted data. Predictions support analysis and should not be the sole basis for customer decisions.")
