"""Churn prediction API. Run from the project root: uvicorn api.main:app --reload"""
import io
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile

from api.schemas import BatchResponse, Customer, PredictionResponse
from src.service import ChurnService

MAX_BYTES = 5 * 1024 * 1024     # 5 MB upload limit
MAX_ROWS = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ChurnService()      # load model and SHAP explainer once
    yield


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


def get_service(request: Request) -> ChurnService:
    return request.app.state.service


@app.get("/health")
def health(svc: ChurnService = Depends(get_service)):
    return svc.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer, svc: ChurnService = Depends(get_service)):
    df = pd.DataFrame([customer.model_dump()])
    try:
        return svc.predict_one(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(
    file: UploadFile = File(...),
    explain_top: int = Query(10, ge=0, le=100,
                             description="Add reasons and actions for the N highest-risk customers"),
    svc: ChurnService = Depends(get_service),
):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 5 MB")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as CSV")
    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows")
    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Maximum {MAX_ROWS} rows per upload")

    try:
        ranked = svc.predict_batch(df, explain_top=explain_top)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    counts = ranked["risk_level"].value_counts().to_dict()
    return {
        "total_customers": len(ranked),
        "counts_by_risk_level": {t: int(counts.get(t, 0)) for t in ("High", "Medium", "Low")},
        "score_is_calibrated": svc.cfg["score_is_calibrated"],
        "customers": ranked.to_dict("records"),
    }