from fastapi.testclient import TestClient
from src.app import app, PREDICTION_THRESHOLD
import pytest

client = TestClient(app)

# Standard payload for reuse
VALID_PAYLOAD = {
    "age": 40,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

class TestPredictEndpoint:
    """Functional tests for the prediction API."""

    def test_predict_happy_path(self):
        """Verify a standard valid request works."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        
        data = response.json()
        # Structural checks
        assert "prediction" in data
        assert "probability" in data
        assert "threshold_used" in data
        assert "model_version" in data
        
        # Value sanity checks
        assert data["prediction"] in ["yes", "no"]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["model_version"] == "v1"

    def test_predict_feature_engineering_bridge(self):
        """
        Verify the 'pdays' logic in app.py. 
        We don't check probability values (model specific), 
        but we check that the request completes successfully for both cases.
        """
        # Case 1: Never contacted (pdays = -1)
        payload_new = VALID_PAYLOAD.copy()
        payload_new["pdays"] = -1
        
        # Case 2: Contacted before (pdays > 0)
        payload_returning = VALID_PAYLOAD.copy()
        payload_returning["pdays"] = 90
        payload_returning["previous"] = 2
        
        resp_1 = client.post("/predict", json=payload_new)
        resp_2 = client.post("/predict", json=payload_returning)
        
        assert resp_1.status_code == 200
        assert resp_2.status_code == 200

    def test_threshold_logic_adherence(self):
        """
        Verify the API strictly follows the loaded threshold.
        This ensures the logic 'if prob >= threshold' is mathematically correct.
        """
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        
        prob = data["probability"]
        threshold = data["threshold_used"]
        pred = data["prediction"]

        # Ensure the API is using the threshold defined in the app
        assert threshold == PREDICTION_THRESHOLD
        
        # Verify the math logic
        expected_pred = "yes" if prob >= threshold else "no"
        assert pred == expected_pred

class TestSchemaValidation:
    """Tests for Pydantic validation (The 'Contract' tests)."""

    def test_invalid_category(self):
        payload = VALID_PAYLOAD.copy()
        payload["job"] = "SpaceMarine" # Invalid
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "job" in response.text

    def test_invalid_data_type(self):
        payload = VALID_PAYLOAD.copy()
        payload["age"] = "Thirty-Five" # Should be int
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_missing_field(self):
        payload = VALID_PAYLOAD.copy()
        del payload["education"]
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_outlier_inputs_handled(self):
        """Ensure Scalers handle extreme values without crashing."""
        payload = VALID_PAYLOAD.copy()
        payload["balance"] = 1_000_000 # Dr. Evil money
        payload["age"] = 99
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200