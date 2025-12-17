"""
Configuration module for the term deposit prediction project.

This module provides declarative path definitions and a configuration loader
that must be explicitly called during application startup. This design:
- Avoids side effects on import (no file I/O or directory creation)
- Makes dependencies explicit and testable
- Allows for easy mocking in unit tests
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# --- Declarative Path Definitions (no side effects) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCHEMA_PATH = BASE_DIR / "src" / "schema_contract.json"


@dataclass
class SchemaConfig:
    """Immutable configuration loaded from schema contract."""
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    id_col: str
    group_col: str
    valid_categories: Dict[str, List[str]] = field(default_factory=dict)


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass


# Global config holder - None until explicitly initialized
_schema_config: Optional[SchemaConfig] = None


def initialize_directories() -> None:
    """
    Create required directories. Call explicitly during application startup.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)


def load_schema_config(schema_path: Path = SCHEMA_PATH) -> SchemaConfig:
    """
    Load and validate schema configuration from JSON file.

    Args:
        schema_path: Path to schema contract JSON file

    Returns:
        SchemaConfig instance with validated configuration

    Raises:
        ConfigurationError: If schema file is missing or invalid
    """
    global _schema_config

    if not schema_path.exists():
        raise ConfigurationError(f"Schema file not found: {schema_path}")

    try:
        with open(schema_path) as f:
            schema_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in schema file: {e}")

    required_keys = ["numeric_features", "categorical_features", "target", "id_col", "group_col"]
    missing_keys = [k for k in required_keys if k not in schema_data]
    if missing_keys:
        raise ConfigurationError(f"Missing required schema keys: {missing_keys}")

    _schema_config = SchemaConfig(
        numeric_features=schema_data["numeric_features"],
        categorical_features=schema_data["categorical_features"],
        target=schema_data["target"],
        id_col=schema_data["id_col"],
        group_col=schema_data["group_col"],
        valid_categories=schema_data.get("valid_categories", {})
    )

    return _schema_config


def get_schema_config() -> SchemaConfig:
    """
    Get the loaded schema configuration.

    Returns:
        SchemaConfig instance

    Raises:
        ConfigurationError: If configuration has not been initialized
    """
    if _schema_config is None:
        raise ConfigurationError(
            "Configuration not initialized. Call load_schema_config() during startup."
        )
    return _schema_config


def reset_config() -> None:
    """Reset configuration state. Useful for testing."""
    global _schema_config
    _schema_config = None


# --- Convenience accessors (lazy loading with clear error messages) ---
def get_numeric_features() -> List[str]:
    """Get numeric feature column names."""
    return get_schema_config().numeric_features


def get_categorical_features() -> List[str]:
    """Get categorical feature column names."""
    return get_schema_config().categorical_features


def get_target() -> str:
    """Get target column name."""
    return get_schema_config().target


def get_id_col() -> str:
    """Get ID column name."""
    return get_schema_config().id_col


def get_group_col() -> str:
    """Get group column name."""
    return get_schema_config().group_col


def get_valid_categories() -> Dict[str, List[str]]:
    """Get valid categories for categorical features."""
    return get_schema_config().valid_categories
