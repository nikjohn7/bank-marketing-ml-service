"""
Training pipeline for term deposit prediction model.

This module decomposes the training workflow into single-responsibility functions
for better testability, maintainability, and reusability.
"""
import pandas as pd
import polars as pl
import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)

from src import config
from src.features import apply_feature_engineering


# --- Data Loading ---

def load_data(path: Path) -> pd.DataFrame:
    """
    Load dataset from parquet file.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame with loaded data
    """
    df = pl.read_parquet(path)
    print(f"Loaded data shape: {df.shape}")
    return df.to_pandas()


# --- Data Splitting ---

def split_data(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    group_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets using grouped shuffle split.

    This ensures that all records from the same group (e.g., customer profile)
    stay in either train or test set, preventing data leakage.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        id_col: Name of ID column to drop
        group_col: Name of column to group by for splitting
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))

    X = df.drop(columns=[target_col, id_col, group_col])
    y = df[target_col]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# --- Preprocessing Pipeline ---

def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    Uses RobustScaler for balance (handles outliers) and StandardScaler for
    other numeric features. OneHotEncoder for categorical features.

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    numeric_transformer = ColumnTransformer([
        ("balance_robust", RobustScaler(), ["balance"]),
        ("other_standard", StandardScaler(), [c for c in numeric_features if c != "balance"])
    ], verbose_feature_names_out=False)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=False
    )

    return preprocessor


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    model_params: Dict[str, Any]
) -> Pipeline:
    """
    Build complete sklearn pipeline with preprocessor and classifier.

    Args:
        preprocessor: ColumnTransformer for preprocessing
        model_params: Hyperparameters for RandomForestClassifier

    Returns:
        Complete sklearn Pipeline
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**model_params))
    ])
    return pipeline


# --- Model Evaluation ---

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate model performance with standard metrics.

    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1_default = f1_score(y_test, y_pred)

    print(f"\nDefault Model Performance (Threshold=0.5):")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 Score: {f1_default:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return {
        "roc_auc": float(auc),
        "f1_default": float(f1_default),
        "y_prob": y_prob
    }


def find_optimal_threshold(y_test: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Find optimal probability threshold based on F1 score.

    Args:
        y_test: True labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary with optimal threshold and related metrics
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # Avoid division by zero
    numerator = 2 * precision * recall
    denominator = precision + recall
    fscore = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)

    ix = np.argmax(fscore)
    best_thresh = thresholds[ix]
    best_f1 = fscore[ix]

    print("--- Threshold Analysis ---")
    print(f"Best Threshold: {best_thresh:.4f} (Max F1: {best_f1:.4f})")
    print(f"At this threshold -> Recall: {recall[ix]:.4f}, Precision: {precision[ix]:.4f}")

    return {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "recall_at_best": float(recall[ix]),
        "precision_at_best": float(precision[ix])
    }


# --- Feature Importance ---

def get_feature_names(column_transformer: ColumnTransformer) -> List[str]:
    """
    Extract feature names from fitted ColumnTransformer.

    Args:
        column_transformer: Fitted ColumnTransformer

    Returns:
        List of feature names
    """
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
            output_features.extend(pipe.get_feature_names_out(features))
        else:
            output_features.extend(features)
    return output_features


def analyze_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract and display feature importance from trained model.

    Args:
        pipeline: Trained sklearn pipeline with RandomForest classifier

    Returns:
        DataFrame with feature names and importance scores
    """
    try:
        feature_names = get_feature_names(pipeline.named_steps['preprocessor'])
        importances = pipeline.named_steps['classifier'].feature_importances_
        feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        top_features = feat_df.sort_values(by='importance', ascending=False).head(10)

        print("\nTop 10 Important Features:")
        print(top_features)
        return feat_df
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame()


# --- Artifact Saving ---

def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    threshold_info: Dict[str, float],
    artifacts_dir: Path,
    data_source: str
) -> Path:
    """
    Save trained model and training metadata.

    Args:
        pipeline: Trained sklearn pipeline
        metrics: Evaluation metrics dictionary
        params: Model hyperparameters
        threshold_info: Optimal threshold analysis results
        artifacts_dir: Directory to save artifacts
        data_source: Path to training data (for provenance)

    Returns:
        Path to saved model file
    """
    model_path = artifacts_dir / "model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": params,
        "metrics": metrics,
        "threshold_analysis": threshold_info,
        "data_source": data_source
    }

    metadata_path = artifacts_dir / "training_run.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    return model_path


# --- Main Training Function ---

def train_model() -> None:
    """
    Main training pipeline orchestration function.

    This function:
    1. Initializes configuration
    2. Loads and preprocesses data
    3. Applies feature engineering
    4. Builds and trains the model
    5. Evaluates and saves artifacts
    """
    print("--- Starting Training Pipeline ---")

    # Initialize configuration (explicit startup)
    config.initialize_directories()
    schema = config.load_schema_config()

    # Hyperparameters
    rf_params = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    # Load data
    data_path = config.DATA_DIR / "dataset_prepared.parquet"
    df = load_data(data_path)

    # Apply feature engineering (shared with inference)
    df = apply_feature_engineering(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_col=schema.target,
        id_col=schema.id_col,
        group_col=schema.group_col
    )

    # Build pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_features=schema.numeric_features,
        categorical_features=schema.categorical_features
    )
    pipeline = build_model_pipeline(preprocessor, rf_params)

    # Train
    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    eval_results = evaluate_model(pipeline, X_test, y_test)

    # Threshold analysis
    threshold_info = find_optimal_threshold(y_test, eval_results["y_prob"])

    # Feature importance
    analyze_feature_importance(pipeline)

    # Save artifacts
    metrics = {
        "roc_auc": eval_results["roc_auc"],
        "f1_default": eval_results["f1_default"]
    }
    save_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        params=rf_params,
        threshold_info=threshold_info,
        artifacts_dir=config.ARTIFACTS_DIR,
        data_source=str(data_path)
    )


if __name__ == "__main__":
    train_model()
