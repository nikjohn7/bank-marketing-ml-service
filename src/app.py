import joblib
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model
from typing import Literal, Optional
from src import config

# Initialize App
app = FastAPI(title="Term Deposit Prediction API", version="1.0.0")

# Load Resources on Startup
# We load the schema to enforce types and the model for prediction
with open(config.SCHEMA_PATH, "r") as f:
    SCHEMA_CONTRACT = json.load(f)

PIPELINE = joblib.load(config.ARTIFACTS_DIR / "model_pipeline.joblib")

# --- Dynamic Pydantic Model ---
# Since we have a strict schema contract, let's create the Input Model based on it.
# This ensures that if the schema changes, the API validation updates automatically.
# (Alternatively, you can write the class manually, but this is "Senior" level dynamism)

def create_input_model(schema):
    fields = {}
    
    # Valid categories from schema
    valid_cats = schema.get("valid_categories", {})
    
    # Add Numeric Fields
    for col in ["age", "balance", "day", "campaign", "previous", "pdays"]: # pdays needed for engineering
        fields[col] = (int, Field(..., description=f"Numeric value for {col}"))
        
    # Add Categorical Fields
    for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:
        # Create a Literal type based on valid categories in schema
        if col in valid_cats:
            # We add "unknown" explicitly if missing, though EDA says it's there
            options = tuple(valid_cats[col])
            fields[col] = (Literal[options], Field(..., description=f"Categorical options: {options}"))
        else:
            fields[col] = (str, Field(..., description="String category"))
            
    return create_model("CustomerData", **fields)

CustomerInput = create_input_model(SCHEMA_CONTRACT)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str = "v1"

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": PIPELINE is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    try:
        # 1. Convert Pydantic to Pandas DataFrame
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])
        
        # 2. On-the-fly Feature Engineering (The Bridge)
        # The model expects 'was_contacted_before', but user sends 'pdays'
        # Logic from EDA: pdays == -1 means not contacted previously
        df["was_contacted_before"] = df["pdays"].apply(lambda x: "No" if x == -1 else "Yes")
        
        # We don't drop pdays here because the Pipeline ignores columns not in its Transformer
        # But for cleanliness, we could drop it.
        
        # 3. Inference
        # Pipeline handles Scaling and OneHotEncoding automatically
        prediction_cls = PIPELINE.predict(df)[0]  # 0 or 1
        prediction_prob = PIPELINE.predict_proba(df)[0][1] # Probability of 1
        
        # 4. Map output
        result_label = "yes" if prediction_cls == 1 else "no"
        
        return {
            "prediction": result_label,
            "probability": round(prediction_prob, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))