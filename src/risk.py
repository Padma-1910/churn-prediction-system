"""Risk tiers, threshold and action rules, loaded from the saved model metadata."""
import json


def load_config(path: str = "models/model_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def risk_tier(score: float, cfg: dict) -> str:
    if score >= cfg["high_min"]:
        return "High"
    if score >= cfg["low_max"]:
        return "Medium"
    return "Low"


def predict_label(score: float, cfg: dict) -> str:
    return "Yes" if score >= cfg["threshold"] else "No"


# Driver -> (business reason, suggested action). A rule fires only when the model says the
# feature RAISES this customer's risk AND the customer's value matches the condition.
# Thresholds (12 months, $70) are starting points: set them from your own EDA.
ACTION_RULES = {
    "Contract": lambda v: ("Month-to-month contract",
        "Offer a discount to move to a 1- or 2-year contract") if v == "Month-to-month" else None,
    "tenure": lambda v: ("Short tenure (first year)",
        "Schedule an onboarding check-in call and welcome offer") if v <= 12 else None,
    "InternetService": lambda v: ("Fiber optic service",
        "Review price and service quality; offer a plan check-up") if v == "Fiber optic" else None,
    "TechSupport": lambda v: ("No tech support",
        "Offer a free tech-support trial") if v == "No" else None,
    "OnlineSecurity": lambda v: ("No online security",
        "Offer a free online-security trial") if v == "No" else None,
    "PaymentMethod": lambda v: ("Pays by electronic check",
        "Offer a small incentive to switch to auto-pay") if v == "Electronic check" else None,
    "MonthlyCharges": lambda v: ("High monthly charges",
        "Review the plan; offer a bundle or loyalty discount") if v >= 70 else None,
    "NumAddOns": lambda v: ("Few add-on services",
        "Offer a bundled add-on package") if v <= 1 else None,
}

DEFAULT_ACTION = "Offer a suitable retention plan"


def suggest_action(drivers):
    """drivers = explain_customer(...)['increasing'] -> (reasons, actions)."""
    reasons, actions = [], []
    for feature, value, _ in drivers:
        rule = ACTION_RULES.get(feature)
        out = rule(value) if rule else None
        if out:
            reasons.append(out[0])
            actions.append(out[1])
        else:
            reasons.append(f"{feature} = {value}")     # explained, but no action attached
    return reasons, (actions or [DEFAULT_ACTION])