import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCHEMA_PATH = BASE_DIR / "src" / "schema_contract.json"

# Ensure dirs exist 
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

with open(SCHEMA_PATH) as f:
    _schema = json.load(f)

NUMERIC_FEATURES = _schema["numeric_features"]
CATEGORICAL_FEATURES = _schema["categorical_features"]
TARGET = _schema["target"]
ID_COL = _schema["id_col"]
GROUP_COL = _schema["group_col"]
