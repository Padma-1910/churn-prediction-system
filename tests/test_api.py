import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

CUSTOMER = {
    "customerID": "TEST-001", "gender": "Female", "SeniorCitizen": 0,
    "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70, "TotalCharges": 212.10,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # "with" is required so startup loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_expected_fields(client):
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "TEST-001"
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert body["prediction"] in ("Yes", "No")
    assert body["suggested_actions"]


def test_invalid_category_is_rejected(client):
    bad = {**CUSTOMER, "Contract": "Monthly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_blank_total_charges_is_accepted(client):
    r = client.post("/predict", json={**CUSTOMER, "tenure": 0, "TotalCharges": None})
    assert r.status_code == 200


def test_batch_is_ranked_by_risk(client):
    df = pd.read_csv("data/telco_churn.csv").head(60)
    r = client.post("/predict-batch?explain_top=3",
                    files={"file": ("batch.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    scores = [c["risk_score"] for c in body["customers"]]
    assert body["total_customers"] == 60
    assert scores == sorted(scores, reverse=True)
    assert body["customers"][0]["top_reasons"] != ""


def test_batch_missing_column_is_rejected(client):
    df = pd.read_csv("data/telco_churn.csv").head(5).drop(columns=["Contract"])
    r = client.post("/predict-batch",
                    files={"file": ("batch.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 422 and "Contract" in r.json()["detail"]


def test_non_csv_is_rejected(client):
    r = client.post("/predict-batch", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400