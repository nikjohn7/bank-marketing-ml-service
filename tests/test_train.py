import joblib
import pandas as pd
import pytest
from src import config

def test_pipeline_exists():
    assert (config.ARTIFACTS_DIR / "model_pipeline.joblib").exists()

def test_pipeline_prediction():
    pipeline = joblib.load(config.ARTIFACTS_DIR / "model_pipeline.joblib")
    
    # Create a minimal valid dataframe matching training schema
    # Note: Must include 'was_contacted_before' because this test hits the pipeline directly,
    # skipping the app.py transformation logic.
    data = pd.DataFrame([{
        "age": 40, "balance": 1000, "day": 1, "campaign": 1, "previous": 0,
        "job": "management", "marital": "married", "education": "tertiary",
        "default": "no", "housing": "yes", "loan": "no", "contact": "unknown",
        "month": "jan", "poutcome": "unknown", "was_contacted_before": "No"
    }])
    
    pred = pipeline.predict(data)
    assert len(pred) == 1