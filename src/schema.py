"""
Schema management module for term deposit prediction.

Instead of relying solely on a manually maintained JSON file, this module provides
tools to:
1. Generate schema from training data (source of truth)
2. Validate schema consistency
3. Update the schema contract file when data changes

This approach reduces the risk of schema drift by making the data itself
the primary source of truth, while still maintaining a serialized contract
for the inference pipeline.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataSchema:
    """Schema definition derived from data."""
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    id_col: str
    group_col: str
    valid_categories: Dict[str, List[str]]
    data_hash: Optional[str] = None  # Hash of source data for drift detection


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


def compute_data_hash(df) -> str:
    """
    Compute a hash of the dataframe structure for drift detection.

    This doesn't hash all the data (which would be slow), but rather
    the structure: column names, dtypes, and sample of unique values.
    """
    import pandas as pd

    structure_str = f"columns:{sorted(df.columns.tolist())}"
    structure_str += f"|dtypes:{df.dtypes.to_dict()}"

    # Add sample of unique values for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = sorted(df[col].dropna().unique().tolist())[:20]
        structure_str += f"|{col}:{unique_vals}"

    return hashlib.md5(structure_str.encode()).hexdigest()[:16]


def generate_schema_from_data(
    df,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    id_col: str,
    group_col: str
) -> DataSchema:
    """
    Generate schema from actual training data.

    This ensures the schema reflects the true state of the data,
    including all valid category values.

    Args:
        df: Training DataFrame
        numeric_cols: List of numeric feature column names
        categorical_cols: List of categorical feature column names
        target_col: Target column name
        id_col: ID column name
        group_col: Grouping column name

    Returns:
        DataSchema with valid categories extracted from data
    """
    # Extract valid categories from actual data
    valid_categories = {}
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            valid_categories[col] = sorted([str(v) for v in unique_vals])

    return DataSchema(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        target=target_col,
        id_col=id_col,
        group_col=group_col,
        valid_categories=valid_categories,
        data_hash=compute_data_hash(df)
    )


def save_schema(schema: DataSchema, path: Path) -> None:
    """
    Save schema to JSON file.

    Args:
        schema: DataSchema to save
        path: Output path for JSON file
    """
    schema_dict = asdict(schema)
    with open(path, "w") as f:
        json.dump(schema_dict, f, indent=2)


def load_schema(path: Path) -> DataSchema:
    """
    Load schema from JSON file.

    Args:
        path: Path to schema JSON file

    Returns:
        DataSchema instance

    Raises:
        SchemaValidationError: If file is missing or invalid
    """
    if not path.exists():
        raise SchemaValidationError(f"Schema file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON in schema file: {e}")

    required_keys = ["numeric_features", "categorical_features", "target", "id_col", "group_col"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise SchemaValidationError(f"Missing required schema keys: {missing}")

    return DataSchema(
        numeric_features=data["numeric_features"],
        categorical_features=data["categorical_features"],
        target=data["target"],
        id_col=data["id_col"],
        group_col=data["group_col"],
        valid_categories=data.get("valid_categories", {}),
        data_hash=data.get("data_hash")
    )


def validate_dataframe_against_schema(df, schema: DataSchema) -> List[str]:
    """
    Validate that a DataFrame conforms to the expected schema.

    Args:
        df: DataFrame to validate
        schema: Expected schema

    Returns:
        List of validation warnings (empty if all good)

    Raises:
        SchemaValidationError: For critical mismatches
    """
    warnings = []

    # Check required columns exist
    expected_cols = (
        schema.numeric_features +
        schema.categorical_features
    )
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(f"Missing required columns: {missing_cols}")

    # Check for unexpected category values
    for col, valid_vals in schema.valid_categories.items():
        if col in df.columns:
            actual_vals = set(df[col].dropna().unique().tolist())
            valid_set = set(valid_vals)
            unexpected = actual_vals - valid_set
            if unexpected:
                warnings.append(
                    f"Column '{col}' has unexpected values: {unexpected}. "
                    f"Expected one of: {valid_vals}"
                )

    return warnings


def check_schema_drift(current_df, stored_schema: DataSchema) -> Dict[str, Any]:
    """
    Check for schema drift between current data and stored schema.

    Args:
        current_df: Current DataFrame
        stored_schema: Previously saved schema

    Returns:
        Dictionary with drift analysis results
    """
    drift_report = {
        "has_drift": False,
        "new_categories": {},
        "missing_categories": {},
        "hash_changed": False
    }

    if stored_schema.data_hash:
        current_hash = compute_data_hash(current_df)
        if current_hash != stored_schema.data_hash:
            drift_report["has_drift"] = True
            drift_report["hash_changed"] = True

    # Check for category drift
    for col, stored_vals in stored_schema.valid_categories.items():
        if col in current_df.columns:
            current_vals = set(str(v) for v in current_df[col].dropna().unique())
            stored_set = set(stored_vals)

            new_vals = current_vals - stored_set
            missing_vals = stored_set - current_vals

            if new_vals:
                drift_report["has_drift"] = True
                drift_report["new_categories"][col] = list(new_vals)

            if missing_vals:
                drift_report["missing_categories"][col] = list(missing_vals)

    return drift_report
