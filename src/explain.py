"""SHAP helpers: explain a saved churn pipeline in terms of the original features."""
import numpy as np
import pandas as pd
import shap

from src.preprocess import clean_and_engineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

ORIGINAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_parts(pipe):
    """Split a saved pipeline into (preprocessor, model, encoded feature names)."""
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    raw_names = prep.named_steps["encode_scale"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in raw_names]       # drop "num__" / "cat__"
    return prep, model, names


def group_of(encoded_name: str) -> str:
    """Map 'Contract_Month-to-month' back to 'Contract'."""
    matches = [f for f in ORIGINAL_FEATURES
               if encoded_name == f or encoded_name.startswith(f + "_")]
    return max(matches, key=len)


def make_explainer(model, X_background=None):
    if model.__class__.__name__ == "LogisticRegression":
        return shap.LinearExplainer(model, X_background)
    return shap.TreeExplainer(model)


def shap_matrix(explainer, X_transformed):
    """Return (SHAP values for the churn class, base value), whatever the shap version."""
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):          # older shap: one array per class
        sv = sv[1]
    elif sv.ndim == 3:                # newer shap with Random Forest: (rows, features, classes)
        sv = sv[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return np.asarray(sv), base


def group_shap(sv, names):
    """Sum one-hot columns so each ORIGINAL feature has one SHAP value per row."""
    groups = [group_of(n) for n in names]
    return pd.DataFrame(sv, columns=names).T.groupby(groups).sum().T


def explain_customer(pipe, explainer, X_raw, top_n=3, exclude=("gender", "SeniorCitizen")):
    """Top drivers for ONE customer (a one-row raw DataFrame).
    `exclude` only hides features from the displayed reasons; the model still uses them."""
    prep, _, names = get_parts(pipe)
    sv, _ = shap_matrix(explainer, prep.transform(X_raw))
    g = group_shap(sv, names).iloc[0]
    g = g.drop(labels=[c for c in exclude if c in g.index])
    cleaned = clean_and_engineer(X_raw).iloc[0]

    up = g[g > 0].sort_values(ascending=False).head(top_n)
    down = g[g < 0].sort_values().head(top_n)
    return {
        "increasing": [(f, cleaned[f], float(v)) for f, v in up.items()],
        "decreasing": [(f, cleaned[f], float(v)) for f, v in down.items()],
    }