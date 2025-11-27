import pandas as pd
import polars as pl
import joblib
import json
import numpy as np
from datetime import datetime
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

def load_data(path: str) -> pd.DataFrame:
    """Loads the prepared parquet file."""
    df = pl.read_parquet(path)
    print(f"Loaded data shape: {df.shape}")
    return df.to_pandas()


def save_training_metadata(metrics, params, artifacts_dir, threshold_info):
    """Saves training run metadata to JSON."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": params,
        "metrics": metrics,
        "threshold_analysis": threshold_info,
        "data_source": str(config.DATA_DIR / "dataset_prepared.parquet")
    }
    
    out_path = artifacts_dir / "training_run.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {out_path}")

def get_feature_names(column_transformer):
    """Helper to extract feature names from the preprocessor."""
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
             # For Scikit-Learn 1.0+
            output_features.extend(pipe.get_feature_names_out(features))
        else:
            output_features.extend(features)
    return output_features

def train_model():
    print("--- Starting Training Pipeline ---")
    
    # 1. Configuration & Hyperparameters
    rf_params = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    # 2. Load Data
    data_path = config.DATA_DIR / "dataset_prepared.parquet"
    df = load_data(str(data_path))
    
    # 3. Split Data (Grouped by profile_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[config.GROUP_COL]))
    
    X = df.drop(columns=[config.TARGET, config.ID_COL, config.GROUP_COL])
    y = df[config.TARGET]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Define Preprocessing
    numeric_transformer = ColumnTransformer([
        ("balance_robust", RobustScaler(), ["balance"]),
        ("other_standard", StandardScaler(), [c for c in config.NUMERIC_FEATURES if c != "balance"])
    ], verbose_feature_names_out=False)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False
    )

    # 5. Build Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**rf_params))
    ])
    
    # 6. Train
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluate (Standard)
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1_default = f1_score(y_test, y_pred)
    
    print(f"\nDefault Model Performance (Threshold=0.5):")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 Score: {f1_default:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 8. Threshold Analysis (The "Senior" addition)
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    # Avoid division by zero
    numerator = 2 * precision * recall
    denominator = precision + recall
    fscore = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix]
    best_f1 = fscore[ix]
    
    print("--- Threshold Analysis ---")
    print(f"Best Threshold: {best_thresh:.4f} (Max F1: {best_f1:.4f})")
    print(f"At this threshold -> Recall: {recall[ix]:.4f}, Precision: {precision[ix]:.4f}")
    
    threshold_info = {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "recall_at_best": float(recall[ix]),
        "precision_at_best": float(precision[ix])
    }

    # 9. Feature Importance
    try:
        feature_names = get_feature_names(pipeline.named_steps['preprocessor'])
        importances = pipeline.named_steps['classifier'].feature_importances_
        feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        top_features = feat_df.sort_values(by='importance', ascending=False).head(10)
        
        print("\nTop 10 Important Features:")
        print(top_features)
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    # 10. Save Artifacts & Metadata
    model_path = config.ARTIFACTS_DIR / "model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    metrics = {
        "roc_auc": float(auc),
        "f1_default": float(f1_default)
    }
    
    save_training_metadata(metrics, rf_params, config.ARTIFACTS_DIR, threshold_info)
    print(f"Pipeline saved to {model_path}")

if __name__ == "__main__":
    train_model()