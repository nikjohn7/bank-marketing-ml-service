"""
FastAPI application for term deposit prediction.

This module uses FastAPI's lifespan context manager for proper resource management:
- Model is loaded during startup, not at module import time
- If model loading fails, the application fails to start completely
- Resources are properly cleaned up during shutdown
"""
import joblib
import logging
import pandas as pd
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, create_model
from sklearn.pipeline import Pipeline

from src import config
from src.features import apply_feature_engineering, FeatureEngineeringError


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Application State ---
class AppState:
    """Container for application state loaded during startup."""
    pipeline: Pipeline = None
    schema_config: config.SchemaConfig = None
    prediction_threshold: float = 0.22  # Optimized from training analysis


app_state = AppState()


# --- Custom Exception Classes ---
class ModelNotLoadedError(Exception):
    """Raised when model pipeline is not available."""
    pass


class StartupError(Exception):
    """Raised when application startup fails."""
    pass


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle with proper startup/shutdown handling.

    On startup:
    - Loads configuration (explicit, not on import)
    - Loads model pipeline
    - Fails fast if any critical resource is unavailable

    On shutdown:
    - Cleans up resources
    """
    logger.info("Starting application...")

    # Load configuration
    try:
        app_state.schema_config = config.load_schema_config()
        logger.info("Configuration loaded successfully")
    except config.ConfigurationError as e:
        logger.error(f"Failed to load configuration: {e}")
        raise StartupError(f"Configuration error: {e}")

    # Load model pipeline - fail fast if not available
    model_path = config.ARTIFACTS_DIR / "model_pipeline.joblib"
    try:
        app_state.pipeline = joblib.load(model_path)
        logger.info(f"Model pipeline loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model pipeline not found at {model_path}")
        raise StartupError(
            f"Model pipeline not found. Please run training first. "
            f"Expected path: {model_path}"
        )
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}")
        raise StartupError(f"Failed to load model: {e}")

    # Load threshold from training metadata if available
    try:
        import json
        metadata_path = config.ARTIFACTS_DIR / "training_run.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            threshold = metadata.get("threshold_analysis", {}).get("best_threshold")
            if threshold:
                app_state.prediction_threshold = threshold
                logger.info(f"Using optimized threshold: {threshold}")
    except Exception as e:
        logger.warning(f"Could not load threshold from metadata: {e}")

    logger.info("Application startup complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")
    app_state.pipeline = None
    app_state.schema_config = None
    config.reset_config()
    logger.info("Shutdown complete")


# --- Initialize App with Lifespan ---
app = FastAPI(
    title="Term Deposit Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


# --- Dynamic Pydantic Model ---
def create_input_model():
    """
    Creates a Pydantic model dynamically from the schema contract.
    This ensures API validation stays in sync with training data categories.

    Note: This is called after startup when schema is loaded.
    """
    if app_state.schema_config is None:
        raise RuntimeError("Schema not loaded. Ensure app has started.")

    valid_cats = app_state.schema_config.valid_categories
    fields = {}

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


# --- Response Models ---
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold_used: float
    model_version: str = "v1"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- Global Exception Handler ---
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Invalid input data"}
    )


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "ok" if app_state.pipeline is not None else "degraded",
        "model_loaded": app_state.pipeline is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    """
    Predict whether a customer will subscribe to a term deposit.

    Uses optimized probability threshold based on F1-score analysis
    during training. This captures more potential subscribers compared to
    the default 0.5 threshold.
    """
    # Model is guaranteed to be loaded due to lifespan startup checks
    # No need for None checks here

    try:
        # Get and validate input data
        body = await request.json()

        # Create input model dynamically based on loaded schema
        CustomerInput = create_input_model()
        data = CustomerInput(**body)

        # Convert Pydantic model to DataFrame
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])

        logger.info(f"Prediction request: job={data.job}, age={data.age}, balance={data.balance}")

    except ValidationError as e:
        logger.warning(f"Input validation failed: {e}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Failed to parse input data: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse input data: {str(e)}"
        )

    try:
        # Feature Engineering (shared with training via features module)
        df = apply_feature_engineering(df)

    except FeatureEngineeringError as e:
        logger.error(f"Feature engineering failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error during feature transformation"
        )

    try:
        # Model Inference
        prediction_prob = app_state.pipeline.predict_proba(df)[0][1]

        # Apply optimized threshold instead of default 0.5
        prediction_cls = 1 if prediction_prob >= app_state.prediction_threshold else 0
        result_label = "yes" if prediction_cls == 1 else "no"

        logger.info(f"Prediction complete: {result_label} (prob={prediction_prob:.4f})")

        return {
            "prediction": result_label,
            "probability": round(prediction_prob, 4),
            "threshold_used": app_state.prediction_threshold,
            "model_version": "v1"
        }

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please check input data format."
        )
