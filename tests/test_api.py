from fastapi.testclient import TestClient
from src.app import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_customer():
    # A customer who should likely say "no" (low balance, previous=0)
    payload = {
        "age": 30,
        "job": "technician",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 100,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,       # Logic check: Should map to was_contacted_before=No
        "previous": 0,
        "poutcome": "unknown"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["yes", "no"]

def test_predict_schema_validation_error():
    # Sending "CEO" as job, which is not in the valid categories from schema
    payload = {
        "age": 30,
        "job": "CEO", # Invalid
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 100,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity