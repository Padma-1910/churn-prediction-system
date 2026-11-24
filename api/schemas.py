"""Request and response models (Pydantic v2)."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70, "TotalCharges": 212.10}})

    customerID: str = "unknown"
    gender: Optional[Literal["Male", "Female"]] = None
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    customer_id: str
    risk_score: float
    score_is_calibrated: bool
    risk_level: Literal["Low", "Medium", "High"]
    prediction: Literal["Yes", "No"]
    top_reasons: list[str]
    protective_factors: list[str]
    suggested_actions: list[str]


class BatchRow(BaseModel):
    rank: int
    customer_id: str
    risk_score: float
    risk_level: Literal["Low", "Medium", "High"]
    prediction: Literal["Yes", "No"]
    top_reasons: str
    suggested_action: str


class BatchResponse(BaseModel):
    total_customers: int
    counts_by_risk_level: dict[str, int]
    score_is_calibrated: bool
    customers: list[BatchRow]