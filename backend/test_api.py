from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_contract():
    response = client.post("/predict", json={
        "age": 35, "gender": "Male", "marital_status": "Married", "dependents": 1,
        "education": "Graduate", "applicant_income": 15000, "coapplicant_income": 4000,
        "savings": 30000, "employment_status": "Salaried", "employer_category": "Private",
        "credit_score": 760, "existing_loans": 1, "dti_ratio": 24, "loan_amount": 50000,
        "loan_term": 60, "loan_purpose": "Home", "collateral_value": 70000,
        "property_area": "Urban",
    })
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["approved"], bool)
    assert 0 <= body["confidence"] <= 100
    assert body["risk_level"] in {"Low", "Medium", "High"}
