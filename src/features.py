"""
Feature engineering module for term deposit prediction.

This module provides a single source of truth for all feature transformations,
ensuring consistency between training and inference pipelines.

Why this matters:
- Train-serve skew occurs when features are computed differently during training vs inference
- Having feature logic in multiple places (train.py, app.py) creates maintenance burden
- A single module makes it easy to add new features and test them in isolation
"""
import pandas as pd
from typing import List, Optional


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""
    pass


def create_was_contacted_before(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'was_contacted_before' feature based on pdays column.

    pdays == -1 means the customer was never contacted in a previous campaign.

    This function handles two scenarios:
    1. Feature already exists (e.g., prepared training data) - no action needed
    2. pdays column exists (e.g., raw data or inference input) - create the feature

    Args:
        df: DataFrame with either 'pdays' column or existing 'was_contacted_before'

    Returns:
        DataFrame with 'was_contacted_before' column

    Raises:
        FeatureEngineeringError: If neither 'pdays' nor 'was_contacted_before' exists
    """
    # If feature already exists, nothing to do
    if "was_contacted_before" in df.columns:
        return df

    # Need pdays to create the feature
    if "pdays" not in df.columns:
        raise FeatureEngineeringError(
            "Column 'pdays' required to create 'was_contacted_before' feature. "
            "Either provide 'pdays' or ensure 'was_contacted_before' already exists."
        )

    df = df.copy()
    df["was_contacted_before"] = df["pdays"].apply(lambda x: "No" if x == -1 else "Yes")
    return df


def apply_feature_engineering(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    This is the main entry point for feature engineering, used by both
    training and inference pipelines.

    Args:
        df: Input DataFrame
        feature_list: Optional list of features to create. If None, creates all.

    Returns:
        DataFrame with engineered features added

    Raises:
        FeatureEngineeringError: If any feature transformation fails
    """
    # Define available feature engineering functions
    feature_functions = {
        "was_contacted_before": create_was_contacted_before,
    }

    features_to_create = feature_list or list(feature_functions.keys())

    for feature_name in features_to_create:
        if feature_name not in feature_functions:
            raise FeatureEngineeringError(f"Unknown feature: {feature_name}")

        try:
            df = feature_functions[feature_name](df)
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create feature '{feature_name}': {e}")

    return df


def validate_features(df: pd.DataFrame, expected_features: List[str]) -> None:
    """
    Validate that all expected features are present in the DataFrame.

    Args:
        df: DataFrame to validate
        expected_features: List of column names that must be present

    Raises:
        FeatureEngineeringError: If any expected feature is missing
    """
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        raise FeatureEngineeringError(f"Missing features: {missing_features}")
