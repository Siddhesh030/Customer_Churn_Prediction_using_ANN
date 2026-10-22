from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RetainAI | Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Glassmorphism UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.55);
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    csv_candidates = [
        ROOT / "data" / "Churn_Modelling.csv",
        ROOT / "Churn_Modelling.csv"
    ]
    for path in csv_candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_artifacts():
    model_path = ROOT / "models" / "churn_ann.keras"
    if not model_path.exists():
        model_path = ROOT / "churn_ann.keras"

    prep_path = ROOT / "models" / "preprocessor.joblib"
    if not prep_path.exists():
        prep_path = ROOT / "preprocessor.joblib"

    if not model_path.exists() or not prep_path.exists():
        return None, None

    model = keras.models.load_model(model_path)
    prep = joblib.load(prep_path)
    return model, prep


@st.cache_data
def load_metrics():
    metrics_path = ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return None


df = load_data()
metrics = load_metrics()
model, prep = load_artifacts()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### ⚡ **RetainAI**")
    st.caption("Enterprise Customer Churn Suite")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Dashboard Overview", "Customer Segmentation", "Predictive Risk Engine", "Model Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### System Status")
    if model is not None and prep is not None:
        st.success("ANN Engine: Active", icon="🟢")
    else:
        st.error("Model Artifacts: Missing", icon="🔴")
        
    if metrics and "threshold" in metrics:
        st.info(f"Optimal Cutoff: **{metrics['threshold']:.2f}**")


# ==========================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================================
if page == "Dashboard Overview":
    st.title("Executive Churn Overview")
    st.caption("Holistic performance metrics, customer attrition rate, and global segment behaviors.")

    if df.empty:
        st.warning("Customer data not found. Please verify the `Churn_Modelling.csv` placement.")
    else:
        # High Level Metrics
        total_customers = len(df)
        churned_customers = int(df["Exited"].sum())
        churn_rate = churned_customers / total_customers
        active_customers = int(df["IsActiveMember"].sum())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Accounts", f"{total_customers:,}")
        kpi2.metric("Churned Accounts", f"{churned_customers:,}", delta=f"{churn_rate:.1%}", delta_color="inverse")
        kpi3.metric("Retention Rate", f"{(1 - churn_rate):.1%}")
        kpi4.metric("Active Member Base", f"{active_customers:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Attrition Breakdown")
            churn_pie = px.pie(
                df,
                names="Exited",
                hole=0.62,
                color="Exited",
                color_discrete_map={0: "#3b82f6", 1: "#f43f5e"},
                template="plotly_dark"
            )
            churn_pie.update_traces(textinfo="percent+label", pull=[0, 0.06])
            churn_pie.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(churn_pie, use_container_width=True)

        with col2:
            st.markdown("##### Geographic Churn Comparison")
            geo_bar = px.histogram(
                df,
                x="Geography",
                color="Exited",
                barmode="group",
                color_discrete_map={0: "#60a5fa", 1: "#fb7185"},
                template="plotly_dark"
            )
            geo_bar.update_layout(
                legend=dict(title="Churn Status", orientation="h", y=1.1, x=0.7),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(geo_bar, use_container_width=True)


# ==========================================================
# PAGE 2: CUSTOMER SEGMENTATION
# ==========================================================
elif page == "Customer Segmentation":
    st.title("Customer Demographics & Patterns")
    st.caption("Slice customer records by country, tenure, and financial behavior.")

    if df.empty:
        st.warning("Customer data not found.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            geos = st.multiselect(
                "Filter by Region",
                options=sorted(df["Geography"].unique()),
                default=sorted(df["Geography"].unique())
            )
        with c2:
            gender_filter = st.selectbox("Gender Focus", ["All"] + sorted(df["Gender"].unique()))

        filtered = df[df["Geography"].isin(geos)]
        if gender_filter != "All":
            filtered = filtered[filtered["Gender"] == gender_filter]

        col1, col2 = st.columns(2)
        with col1:
            age_hist = px.histogram(
                filtered,
                x="Age",
                color="Exited",
                nbins=28,
                title="Age Distribution vs Attrition",
                color_discrete_map={0: "#38bdf8", 1: "#f43f5e"},
                template="plotly_dark"
            )
            age_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(age_hist, use_container_width=True)

        with col2:
            prod_bar = px.histogram(
                filtered,
                x="NumOfProducts",
                color="Exited",
                barmode="group",
                title="Holding Products vs Attrition",
                color_discrete_map={0: "#818cf8", 1: "#fb7185"},
                template="plotly_dark"
            )
            prod_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(prod_bar, use_container_width=True)

        st.markdown("##### Filtered Records")
        display_cols = [c for c in filtered.columns if c not in ["Surname", "CustomerId", "RowNumber"]]
        st.dataframe(filtered[display_cols], use_container_width=True, height=320)


# ==========================================================
# PAGE 3: PREDICTIVE RISK ENGINE
# ==========================================================
elif page == "Predictive Risk Engine":
    st.title("Interactive Risk Simulator")
    st.caption("Generate real-time neural network inference with tailored customer retention advice.")

    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### Input Profile")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                credit = st.slider("Credit Score", 300, 850, 640)
                geo = st.selectbox("Region", ["France", "Germany", "Spain"])
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.slider("Customer Age", 18, 92, 42)
                tenure = st.slider("Tenure with Firm (Years)", 0, 10, 4)
            with c2:
                balance = st.number_input("Current Balance (€)", 0.0, 300000.0, 60000.0, step=2500.0)
                products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
                card = st.selectbox("Has Active Credit Card?", ["Yes", "No"])
                active = st.selectbox("Is Active Member?", ["Yes", "No"])
                salary = st.number_input("Estimated Salary (€)", 0.0, 300000.0, 75000.0, step=2500.0)

            submitted = st.form_submit_button("Run Risk Prediction", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### Risk Analysis & Gauge")
        if submitted:
            if model is None or prep is None:
                st.error("Model artifacts are missing. Run `python train_model.py` to compile the models.")
            else:
                input_df = pd.DataFrame([{
                    "CreditScore": credit,
                    "Geography": geo,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": products,
                    "HasCrCard": 1 if card == "Yes" else 0,
                    "IsActiveMember": 1 if active == "Yes" else 0,
                    "EstimatedSalary": salary
                }])

                transformed = prep.transform(input_df).astype("float32")
                probability = float(model.predict(transformed, verbose=0)[0][0])
                threshold = metrics.get("threshold", 0.5) if metrics else 0.5

                # Gauge indicator
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 38, "color": "#f8fafc"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": "#f43f5e" if probability >= threshold else "#10b981"},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "steps": [
                            {"range": [0, threshold * 100], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [threshold * 100, 100], "color": "rgba(244, 63, 94, 0.25)"}
                        ],
                        "threshold": {
                            "line": {"color": "#f59e0b", "width": 3},
                            "thickness": 0.8,
                            "value": threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=260,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc"},
                    margin=dict(t=20, b=10, l=20, r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                if probability >= threshold:
                    st.error(f"High Churn Risk Detected (Probability: {probability:.1%})")
                    st.markdown("""
                    **Retention Strategy:**
                    - Immediate check-in with dedicated relationship manager.
                    - Offer personalized fee discounts or customized product bundles.
                    - Validate account activity blockers.
                    """)
                else:
                    st.success(f"Customer Retained (Probability: {probability:.1%})")
                    st.markdown("""
                    **Engagement Strategy:**
                    - Standard cross-sell opportunities for loyalty programs.
                    - Routine digital marketing updates.
                    """)
        else:
            st.info("Fill out the customer metrics and click **Run Risk Prediction** to visualize attrition probability.")


# ==========================================================
# PAGE 4: MODEL DIAGNOSTICS
# ==========================================================
elif page == "Model Diagnostics":
    st.title("Held-out Test Performance & Architecture")
    st.caption("Rigorous evaluation criteria from the independent test set.")

    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Technical Metadata")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"""
            - **Decision Threshold:** `{metrics['threshold']:.2f}` (Optimized via Validation F1)
            - **Test Partition Size:** `{metrics['test_samples']:,}` samples
            - **Input Features:** `10 total` (8 Numeric scaled, 2 Categorical one-hot encoded)
            """)
        with t2:
            st.markdown("""
            - **Architecture:** Feedforward ANN (`Dense 32` → `Dropout 0.2` → `Dense 16` → `Dense 1`)
            - **Loss Function:** Binary Crossentropy
            - **Optimizer:** Adam ($\eta=0.001$)
            """)
    else:
        st.info("Execute `python train_model.py` to view model test metrics.")
