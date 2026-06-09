from __future__ import annotations

import io
import sys
import os
import tempfile
from typing import Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from ingestion import Ingest
from preprocessing import Preprocessor
from splitter import SplitClass
from models.trainer import MODEL_REGISTRY, train_model


app = FastAPI(title="ML Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory state (single-user dev mode)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ColumnRole(BaseModel):
    name: str
    role: str          # "feature" | "target" | "ignore"
    dtype_override: Optional[str] = None  # "numeric" | "categorical" | None


class PipelineConfig(BaseModel):
    column_roles: list[ColumnRole]
    encoding: str = "onehot"            # "onehot" | "label"
    scaling: str = "standard"           # "standard" | "minmax" | "none"
    missing_numeric_strategy: str = "mean"
    missing_categorical_strategy: str = "most_frequent"
    test_size: float = 0.2
    stratify: bool = False
    random_state: int = 42


class ModelConfig(BaseModel):
    model_name: str
    params: dict[str, Any] = {}


class TrainRequest(BaseModel):
    models: list[ModelConfig]
    cv_k: int = 5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a CSV file, run ingestion, return schema + data preview.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()

    # Write to a temp file so Ingest can use filepath
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    ingestor = Ingest()
    schema = ingestor.ingest(tmp_path)

    # Store raw DataFrame for later pipeline run
    _state["raw_df"] = schema["data"]
    _state["filepath"] = tmp_path

    # Build preview (first 10 rows as list-of-dicts)
    preview = schema["data"].head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "columns": schema["colnames"],
        "datatypes": schema["datatypes"],
        "null_counts": dict(schema["null counts"]),
        "total_rows": schema["total entries"],
        "preview": preview,
    }


@app.post("/api/pipeline/run")
def run_pipeline(config: PipelineConfig):
    """
    Apply column roles + dtype overrides, then run preprocessing and split.
    Returns split shapes and class distribution of the target.
    """
    if "raw_df" not in _state:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    df: pd.DataFrame = _state["raw_df"].copy()

    # Apply dtype overrides
    for col_role in config.column_roles:
        if col_role.dtype_override and col_role.name in df.columns:
            if col_role.dtype_override == "numeric":
                df[col_role.name] = pd.to_numeric(df[col_role.name], errors="coerce")
            elif col_role.dtype_override == "categorical":
                df[col_role.name] = df[col_role.name].astype(str)

    # Determine target and ignore columns
    target_col = None
    ignore_cols = []
    for col_role in config.column_roles:
        if col_role.role == "target":
            target_col = col_role.name
        elif col_role.role == "ignore":
            ignore_cols.append(col_role.name)

    if target_col is None:
        raise HTTPException(status_code=400, detail="No target column selected.")

    # Drop ignored columns
    df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    # Class distribution BEFORE preprocessing
    class_dist = df[target_col].value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}

    scaling_val = None if config.scaling == "none" else config.scaling

    pp_config = {
        "missing_numeric_strategy": config.missing_numeric_strategy,
        "missing_categorical_strategy": config.missing_categorical_strategy,
        "scaling": scaling_val,
        "encoding": config.encoding,
        "target_column": target_col,
    }

    preprocessor = Preprocessor(config=pp_config)
    df_processed = preprocessor.fit_transform(df)

    split_config = {
        "test_size": config.test_size,
        "random_state": config.random_state,
        "stratify": config.stratify,
        "target_column": target_col,
    }

    splitter = SplitClass(config=split_config)
    split = splitter.split(df_processed)

    # Persist for training step
    _state["split"] = split
    _state["target_col"] = target_col
    _state["preprocessor"] = preprocessor

    shapes = split.shapes()

    return {
        "status": "ok",
        "target_column": target_col,
        "class_distribution": class_dist,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "feature_columns": list(split.X_train.columns),
        "n_features": split.X_train.shape[1],
    }


@app.get("/api/models/registry")
def get_model_registry():
    """Return MODEL_REGISTRY metadata (no class objects) for the frontend."""
    result = {}
    for key, entry in MODEL_REGISTRY.items():
        result[key] = {
            "label": entry["label"],
            "defaults": _jsonify(entry["defaults"]),
            "param_schema": entry["param_schema"],
        }
    return result


@app.post("/api/train")
def train_models(request: TrainRequest):
    """
    Train one or more models on the stored split.
    Returns per-model results plus a comparison summary.
    """
    if "split" not in _state:
        raise HTTPException(status_code=400, detail="Pipeline has not been run yet.")

    split = _state["split"]
    results = []

    for mc in request.models:
        try:
            result = train_model(
                model_name=mc.model_name,
                params=mc.params,
                X_train=split.X_train,
                X_test=split.X_test,
                y_train=split.y_train,
                y_test=split.y_test,
                cv_k=request.cv_k,
            )
            results.append(_jsonify(result))
        except Exception as e:
            results.append({
                "model_name": mc.model_name,
                "error": str(e),
            })

    # Build comparison (only successful results)
    successful = [r for r in results if "error" not in r]
    comparison = []
    best_model = None
    if successful:
        best_f1 = -1.0
        for r in successful:
            comparison.append({
                "label": r["label"],
                "model_name": r["model_name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r.get("roc_auc"),
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
                "training_time_s": r["training_time_s"],
            })
            if r["f1"] > best_f1:
                best_f1 = r["f1"]
                best_model = r["model_name"]

    return {
        "results": results,
        "comparison": comparison,
        "best_model": best_model,
    }
