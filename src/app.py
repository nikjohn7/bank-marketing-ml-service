import joblib
import logging
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, create_model
from typing import Literal
from src import config

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize App
app = FastAPI(title="Term Deposit Prediction API", version="1.0.0")

# --- Constants ---
# Optimized threshold from training analysis (see training_run.json)
# Default sklearn threshold is 0.5, but analysis showed 0.22 maximizes F1
PREDICTION_THRESHOLD = 0.22

# Load Resources on Startup
try:
    with open(config.SCHEMA_PATH, "r") as f:
        SCHEMA_CONTRACT = json.load(f)
    logger.info("Schema contract loaded successfully")
except FileNotFoundError:
    logger.error(f"Schema file not found at {config.SCHEMA_PATH}")
    raise

try:
    PIPELINE = joblib.load(config.ARTIFACTS_DIR / "model_pipeline.joblib")
    logger.info("Model pipeline loaded successfully")
except FileNotFoundError:
    logger.error(f"Model pipeline not found in {config.ARTIFACTS_DIR}")
    PIPELINE = None


# --- Custom Exception Classes ---
class ModelNotLoadedError(Exception):
    """Raised when model pipeline is not available."""
    pass


class FeatureEngineeringError(Exception):
    """Raised when feature transformation fails."""
    pass


# --- Global Exception Handler ---
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Invalid input data"}
    )


# --- Dynamic Pydantic Model ---
def create_input_model(schema):
    """
    Creates a Pydantic model dynamically from the schema contract.
    This ensures API validation stays in sync with training data categories.
    """
    fields = {}
    valid_cats = schema.get("valid_categories", {})
    
    # Numeric Fields
    for col in ["age", "balance", "day", "campaign", "previous", "pdays"]:
        fields[col] = (int, Field(..., description=f"Numeric value for {col}"))
        
    # Categorical Fields with Literal types for strict validation
    for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:
        if col in valid_cats:
            options = tuple(valid_cats[col])
            fields[col] = (Literal[options], Field(..., description=f"One of: {options}"))
        else:
            fields[col] = (str, Field(..., description="String category"))
            
    return create_model("CustomerData", **fields)


CustomerInput = create_input_model(SCHEMA_CONTRACT)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold_used: float = PREDICTION_THRESHOLD
    model_version: str = "v1"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "ok" if PIPELINE is not None else "degraded",
        "model_loaded": PIPELINE is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):
    """
    Predict whether a customer will subscribe to a term deposit.
    
    Uses optimized probability threshold (0.22) based on F1-score analysis
    during training. This captures more potential subscribers compared to
    the default 0.5 threshold.
    """
    # Check model availability
    if PIPELINE is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please ensure model is trained and artifacts exist."
        )
    
    try:
        # Convert Pydantic model to DataFrame
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])
        
        logger.info(f"Prediction request: job={data.job}, age={data.age}, balance={data.balance}")
        
    except Exception as e:
        logger.error(f"Failed to parse input data: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse input data: {str(e)}"
        )
    
    try:
        # Feature Engineering (matches training logic)
        # pdays == -1 means customer was never contacted before
        df["was_contacted_before"] = df["pdays"].apply(lambda x: "No" if x == -1 else "Yes")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error during feature transformation"
        )
    
    try:
        # Model Inference
        prediction_prob = PIPELINE.predict_proba(df)[0][1]
        
        # Apply optimized threshold instead of default 0.5
        prediction_cls = 1 if prediction_prob >= PREDICTION_THRESHOLD else 0
        result_label = "yes" if prediction_cls == 1 else "no"
        
        logger.info(f"Prediction complete: {result_label} (prob={prediction_prob:.4f})")
        
        return {
            "prediction": result_label,
            "probability": round(prediction_prob, 4),
            "threshold_used": PREDICTION_THRESHOLD,
            "model_version": "v1"
        }
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please check input data format."
        )