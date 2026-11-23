"""Shared preprocessing for the churn project (used by notebooks, API and app)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges",
                    "NumAddOns", "AvgMonthlySpend"]

CATEGORICAL_FEATURES = ["Partner", "Dependents",
                        "PhoneService", "MultipleLines", "InternetService",
                        *ADDON_COLS, "Contract", "PaperlessBilling", "PaymentMethod"]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning and feature creation. Learns nothing from the data,
    so it is safe to apply before the train/test split and at prediction time."""
    df = df.copy()

    # TotalCharges: text -> number; blanks (tenure 0, not yet billed) -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # SeniorCitizen 0/1 -> No/Yes so all binary columns look the same
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Merge redundant categories
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    for col in ADDON_COLS:
        df[col] = df[col].replace("No internet service", "No")

    # Engineered features
    df["NumAddOns"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 avoids divide-by-zero
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # drops customerID and anything else
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Cleaning + encoding + scaling, without a model."""
    return Pipeline([
        ("features", FunctionTransformer(clean_and_engineer)),
        ("encode_scale", build_preprocessor()),
    ])


def make_model_pipeline(model) -> Pipeline:
    """Full pipeline: raw customer DataFrame in, churn probability out."""
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", model),
    ])


def load_data(path: str = "data/telco_churn.csv"):
    df = pd.read_csv(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])          # customerID stays in X
    return X, y


def get_train_test_split(path: str = "data/telco_churn.csv", test_size: float = 0.2):
    X, y = load_data(path)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)