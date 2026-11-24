"""Prediction service shared by the FastAPI backend and the Streamlit app."""
import logging
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.explain import explain_customer, get_parts, make_explainer
from src.risk import load_config, predict_label, risk_tier, suggest_action

log = logging.getLogger("churn.service")
ROOT = Path(__file__).resolve().parents[1]

# Raw columns a customer record must contain. customerID and gender are optional.
REQUIRED_COLUMNS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
]
TEXT_COLUMNS = [c for c in REQUIRED_COLUMNS
                if c not in ("SeniorCitizen", "tenure", "MonthlyCharges")]

NO_ACTION = "No retention action needed; keep monitoring"


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw customer DataFrame and fill the optional columns.
    Raises ValueError with a readable message if something is wrong."""
    df = df.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = pd.NA          # the pipeline treats blank as 0

    if "customerID" not in df.columns:
        df["customerID"] = [f"row-{i + 1}" for i in range(len(df))]
    if "gender" not in df.columns:
        df["gender"] = "Unknown"
    df["gender"] = df["gender"].fillna("Unknown")

    for col in ("tenure", "MonthlyCharges"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    if bad:
        raise ValueError(f"{bad} row(s) have empty or non-numeric values in required columns")

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


class ChurnService:
    def __init__(self,
                 model_path=ROOT / "models" / "churn_model.joblib",
                 config_path=ROOT / "models" / "model_config.json",
                 background_path=ROOT / "models" / "background.csv"):
        self.pipe = joblib.load(model_path)
        self.cfg = load_config(str(config_path))

        saved = self.cfg.get("sklearn_version")
        if saved and saved != sklearn.__version__:
            log.warning("Model was saved with scikit-learn %s but %s is installed",
                        saved, sklearn.__version__)

        prep, model, _ = get_parts(self.pipe)
        background = None
        if model.__class__.__name__ == "LogisticRegression":
            # Linear SHAP needs a background sample (about 100 raw training rows)
            background = prep.transform(pd.read_csv(background_path))
        self.explainer = make_explainer(model, background)

    def info(self) -> dict:
        return {"status": "ok",
                "model": self.cfg["model_name"],
                "threshold": self.cfg["threshold"],
                "score_is_calibrated": self.cfg["score_is_calibrated"]}

    def score(self, X: pd.DataFrame):
        return self.pipe.predict_proba(X)[:, 1]

    def _explain_row(self, row: pd.DataFrame, score: float) -> dict:
        tier = risk_tier(score, self.cfg)
        drivers = explain_customer(self.pipe, self.explainer, row, top_n=3)
        reasons, actions = suggest_action(drivers["increasing"])
        if tier == "Low":
            actions = [NO_ACTION]
        return {
            "customer_id": str(row["customerID"].iloc[0]),
            "risk_score": round(score, 4),
            "score_is_calibrated": self.cfg["score_is_calibrated"],
            "risk_level": tier,
            "prediction": predict_label(score, self.cfg),
            "top_reasons": reasons,
            "protective_factors": [f"{f} = {v}" for f, v, _ in drivers["decreasing"]],
            "suggested_actions": actions,
        }

    def predict_one(self, df: pd.DataFrame) -> dict:
        X = prepare_frame(df)
        score = float(self.score(X)[0])
        return self._explain_row(X.iloc[[0]], score)

    def predict_batch(self, df: pd.DataFrame, explain_top: int = 10) -> pd.DataFrame:
        X = prepare_frame(df)
        scores = self.score(X)

        out = pd.DataFrame({
            "customer_id": X["customerID"].astype(str),
            "risk_score": scores.round(4),
            "risk_level": [risk_tier(s, self.cfg) for s in scores],
            "prediction": [predict_label(s, self.cfg) for s in scores],
            "top_reasons": "",
            "suggested_action": "",
        })
        order = out.sort_values("risk_score", ascending=False, kind="stable").index

        # SHAP is slow per row, so only explain the highest-risk customers
        for pos in order[:explain_top]:
            info = self._explain_row(X.iloc[[pos]], float(scores[pos]))
            out.at[pos, "top_reasons"] = "; ".join(info["top_reasons"])
            out.at[pos, "suggested_action"] = " | ".join(info["suggested_actions"])

        out = out.loc[order].reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out