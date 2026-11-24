"""Streamlit frontend. Run from the project root: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))      # lets "from src..." work however the app is launched

import pandas as pd
import streamlit as st

from src.service import ChurnService

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

YES_NO = ["Yes", "No"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"]
ADDONS = [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
          ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
          ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]


# ---------- cached loaders ----------
@st.cache_resource(show_spinner="Loading model...")
def load_service() -> ChurnService:
    return ChurnService()


@st.cache_data
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "telco_churn.csv")
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


def score_label(svc: ChurnService) -> str:
    return "Churn probability" if svc.cfg["score_is_calibrated"] else "Risk score"


def show_risk_banner(level: str, text: str):
    {"High": st.error, "Medium": st.warning, "Low": st.success}[level](text)


# ---------- page 1: single customer ----------
def page_single(svc: ChurnService):
    st.header("Single customer prediction")
    st.caption("Enter a customer's details to see their churn risk, the main reasons "
               "and a suggested retention action.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Account")
        customer_id = st.text_input("Customer ID (optional)", "NEW-001")
        tenure = st.number_input("Tenure (months)", 0, 100, 3)
        contract = st.selectbox("Contract", CONTRACTS)
        paperless = st.selectbox("Paperless billing", YES_NO)
        payment = st.selectbox("Payment method", PAYMENTS)
        monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.70, step=0.5)
        total = st.number_input("Total charges ($, 0 = estimate)", 0.0, 20000.0, 0.0, step=10.0,
                                help="Leave at 0 to estimate it as tenure x monthly charges.")
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", YES_NO)
        multi = st.selectbox("Multiple lines", ["No", "Yes"], disabled=(phone == "No"))
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon_values = {}
        for col, label in ADDONS:
            addon_values[col] = st.selectbox(label, ["No", "Yes"], disabled=(internet == "No"))
    with c3:
        st.subheader("Household")
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    if not st.button("Predict churn risk", type="primary"):
        return

    record = {
        "customerID": customer_id or "NEW-001",
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else multi,
        "InternetService": internet,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total) if total > 0 else float(tenure) * float(monthly),
    }
    for col, _ in ADDONS:
        record[col] = "No internet service" if internet == "No" else addon_values[col]

    try:
        res = svc.predict_one(pd.DataFrame([record]))
    except ValueError as e:
        st.error(str(e))
        return

    st.divider()
    st.subheader(f"Customer {res['customer_id']}")
    show_risk_banner(res["risk_level"], f"{res['risk_level']} risk")

    m1, m2, m3 = st.columns(3)
    m1.metric(score_label(svc), f"{res['risk_score']:.0%}")
    m2.metric("Predicted churn", res["prediction"])
    m3.metric("Risk level", res["risk_level"])
    st.progress(min(max(res["risk_score"], 0.0), 1.0))
    if not res["score_is_calibrated"]:
        st.caption("This is a model risk score used for ranking customers. "
                   "It is not a literal probability of leaving.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top reasons (raise the risk)**")
        for r in res["top_reasons"] or ["No strong risk drivers"]:
            st.write(f"- {r}")
        if res["protective_factors"]:
            st.markdown("**Protective factors (lower the risk)**")
            for p in res["protective_factors"]:
                st.write(f"- {p}")
    with right:
        st.markdown("**Suggested action**")
        for a in res["suggested_actions"]:
            st.write(f"- {a}")
    st.caption("Reasons come from SHAP and describe how the model scores this customer. "
               "Actions are hypotheses to test, not proven causes of churn.")


# ---------- page 2: batch ----------
def page_batch(svc: ChurnService):
    st.header("Batch prediction")
    st.caption("Upload a CSV of customers to get them ranked from highest to lowest risk.")

    sample = ROOT / "data" / "sample_batch.csv"
    if sample.exists():
        st.download_button("Download a sample CSV", sample.read_bytes(),
                           "sample_batch.csv", "text/csv")

    file = st.file_uploader("Customer CSV", type=["csv"])
    explain_top = st.slider("Add reasons and actions for the top N highest-risk customers",
                            0, 50, 10)

    if file is not None and st.button("Run batch prediction", type="primary"):
        try:
            df = pd.read_csv(file)
            if df.empty:
                st.error("The CSV has no rows.")
                return
            if len(df) > 5000:
                st.error("Please upload at most 5,000 rows.")
                return
            with st.spinner("Scoring customers..."):
                st.session_state["batch"] = svc.predict_batch(df, explain_top=explain_top)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception:
            st.error("Could not read or score this file. Check that it is a valid customer CSV.")
            return

    ranked = st.session_state.get("batch")
    if ranked is None:
        return

    counts = ranked["risk_level"].value_counts()
    a, b, c, d = st.columns(4)
    a.metric("Customers", len(ranked))
    b.metric("High risk", int(counts.get("High", 0)))
    c.metric("Medium risk", int(counts.get("Medium", 0)))
    d.metric("Low risk", int(counts.get("Low", 0)))

    levels = st.multiselect("Show risk levels", ["High", "Medium", "Low"],
                            default=["High", "Medium", "Low"])
    view = ranked[ranked["risk_level"].isin(levels)]
    st.dataframe(
        view, hide_index=True,
        column_config={"risk_score": st.column_config.ProgressColumn(
            score_label(svc), min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.download_button("Download ranked results (CSV)",
                       ranked.to_csv(index=False).encode("utf-8"),
                       "churn_ranked.csv", "text/csv")
    st.caption("Reasons and actions are filled in only for the top N customers because "
               "SHAP is slow on free hosting.")


# ---------- page 3: dashboard ----------
def show_image(name: str, caption: str):
    path = ROOT / "reports" / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{name} not found in reports/")


def show_table(name: str, title: str):
    path = ROOT / "reports" / name
    if path.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(path, index_col=0))


def page_dashboard(svc: ChurnService):
    st.header("Dashboard")
    cfg = svc.cfg
    df = load_raw()

    a, b, c, d = st.columns(4)
    a.metric("Customers in dataset", f"{len(df):,}")
    b.metric("Overall churn rate", f"{df['ChurnFlag'].mean():.1%}")
    c.metric("Selected model", cfg["model_name"])
    d.metric("Decision threshold", cfg["threshold"])
    st.caption(f"Risk tiers: Low below {cfg['low_max']}, Medium {cfg['low_max']} to "
               f"{cfg['high_min']}, High from {cfg['high_min']}.")

    st.subheader("Churn rate by segment (%)")
    df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                               labels=["0-12", "13-24", "25-48", "49-72"])
    cols = st.columns(2)
    for i, col in enumerate(["Contract", "InternetService", "PaymentMethod", "TenureGroup"]):
        rate = (df.groupby(col, observed=True)["ChurnFlag"].mean() * 100).round(1)
        with cols[i % 2]:
            st.markdown(f"**By {col}**")
            st.bar_chart(rate)

    st.subheader("Model comparison")
    show_table("model_comparison_cv.csv", "Cross-validation results (training data)")
    show_table("model_comparison_test.csv", "Test-set results")
    t1, t2, t3 = st.tabs(["ROC curves", "SHAP summary", "Feature importance"])
    with t1:
        show_image("roc_curves.png", "ROC curves (test set)")
    with t2:
        show_image("shap_summary.png", "What drives the churn score")
    with t3:
        show_image("shap_importance.png", "Mean |SHAP value| by original feature")

    st.subheader("Key insights")
    st.write("Add your own 5 to 7 insights from Phase 3 and Phase 7 here, using your own numbers.")
    st.caption("Insights are associations found in historical data, not proven causes.")


# ---------- main ----------
try:
    service = load_service()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

st.sidebar.title("Churn Prediction")
page = st.sidebar.radio("Page", ["Single customer", "Batch upload", "Dashboard"])
st.sidebar.caption(f"Model: {service.cfg['model_name']}")
st.sidebar.caption("Demo built on the public IBM Telco dataset.")

{"Single customer": page_single, "Batch upload": page_batch,
 "Dashboard": page_dashboard}[page](service)