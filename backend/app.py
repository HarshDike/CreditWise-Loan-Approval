from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "credit_wise_model.pkl").open("rb") as model_file:
    artifact = pickle.load(model_file)

model = artifact["model"]
scaler = artifact["scaler"]
encoder = artifact["one_hot_encoder"]
feature_columns = artifact["feature_columns"]

app = FastAPI(title="CreditWise Loan Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):
    age: int = Field(ge=18, le=100)
    gender: str
    marital_status: str
    dependents: int = Field(ge=0, le=20)
    education: str
    applicant_income: float = Field(ge=0)
    coapplicant_income: float = Field(ge=0)
    savings: float = Field(ge=0)
    employment_status: str
    employer_category: str
    credit_score: float = Field(ge=300, le=900)
    existing_loans: int = Field(ge=0)
    dti_ratio: float = Field(ge=0, le=100)
    loan_amount: float = Field(gt=0)
    loan_term: int = Field(gt=0)
    loan_purpose: str
    collateral_value: float = Field(ge=0)
    property_area: str


def suggestions(application: LoanApplication, approved: bool) -> list[str]:
    items: list[str] = []
    if application.credit_score < 700:
        items.append("Improve your credit score by making existing loan payments on time.")
    if application.dti_ratio > 40:
        items.append("Reduce your debt-to-income ratio below 40% before applying.")
    if application.savings < application.loan_amount * 0.2:
        items.append("Build savings toward at least 20% of the requested loan amount.")
    if not items:
        items.append("Maintain your current credit and repayment habits.")
        items.append("Keep an emergency fund alongside future loan payments.")
    elif approved:
        items.append("Compare repayment terms before accepting a loan offer.")
    return items[:3]


def model_input(application: LoanApplication) -> pd.DataFrame:
    # Translate display labels into the exact vocabulary used during training.
    employment = {"Self Employed": "Self-employed"}.get(
        application.employment_status, application.employment_status
    )
    purpose = {"Vehicle": "Car", "Other": "Business"}.get(
        application.loan_purpose, application.loan_purpose
    )
    area = {"Semi Urban": "Semiurban"}.get(application.property_area, application.property_area)
    employer = {"Other": "Business"}.get(application.employer_category, application.employer_category)
    marital = "Single" if application.marital_status == "Divorced" else application.marital_status

    categories = pd.DataFrame([{
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Gender": application.gender,
        "Employer_Category": employer,
    }])
    encoded = encoder.transform(categories)
    encoded_frame = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # LabelEncoder in the training notebook encoded Graduate=0, Not Graduate=1.
    education_level = 0 if application.education in {"Graduate", "Post Graduate", "Doctorate"} else 1
    dti = application.dti_ratio / 100
    values = {
        "Applicant_Income": application.applicant_income,
        "Coapplicant_Income": application.coapplicant_income,
        "Age": application.age,
        "Dependents": application.dependents,
        "Existing_Loans": application.existing_loans,
        "Savings": application.savings,
        "Collateral_Value": application.collateral_value,
        "Loan_Amount": application.loan_amount,
        "Loan_Term": application.loan_term,
        "Education_Level": education_level,
        "DTI_Ratio_sq": dti**2,
        "Credit_Score_sq": application.credit_score**2,
    }
    values.update(encoded_frame.iloc[0].to_dict())
    return pd.DataFrame([[values.get(column, 0.0) for column in feature_columns]], columns=feature_columns)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": artifact["model_name"]}


@app.post("/predict")
def predict(application: LoanApplication) -> dict:
    try:
        features = model_input(application)
        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])
        probabilities = model.predict_proba(scaled)[0]
        confidence = int(round(float(np.max(probabilities)) * 100))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not process this application.") from exc

    approved = prediction == 1
    rejection_probability = float(probabilities[0])
    risk = "Low" if rejection_probability < 0.3 else "Medium" if rejection_probability < 0.65 else "High"
    return {
        "approved": approved,
        "confidence": confidence,
        "risk_level": risk,
        "credit_score": application.credit_score,
        "dti_ratio": application.dti_ratio,
        "suggestions": suggestions(application, approved),
    }
