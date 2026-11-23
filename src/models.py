"""Model definitions for the comparison (all with imbalance handling)."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_models(scale_pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE),            # weighted via sample_weight at fit time
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE),
    }