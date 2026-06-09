"""
trainer.py
----------
Centralised model registry and training entry-point.

Exposes:
  MODEL_REGISTRY  – dict mapping model names to their metadata + class
  train_model()   – single function the UI / API calls to train any model
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from models.logisticregression import LogisticRegressionModel
from models.decisiontree import DecisionTree
from models.randomforest import RandomForest
from models.knn import KNN


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "class": LogisticRegressionModel,
        "defaults": LogisticRegressionModel.DEFAULTS,
        "param_schema": {
            "C": {
                "type": "float",
                "min": 0.01,
                "max": 100.0,
                "step": 0.01,
                "description": "Inverse of regularisation strength. Smaller = stronger regularisation.",
            },
            "max_iter": {
                "type": "int",
                "min": 50,
                "max": 1000,
                "step": 50,
                "description": "Maximum number of iterations for the solver to converge.",
            },
            "solver": {
                "type": "select",
                "options": ["lbfgs", "saga", "liblinear"],
                "description": "Optimisation algorithm. lbfgs works well for most cases.",
            },
            "penalty": {
                "type": "select",
                "options": ["l2", "l1"],
                "description": "Regularisation penalty type.",
            },
        },
    },
    "decision_tree": {
        "label": "Decision Tree",
        "class": DecisionTree,
        "defaults": DecisionTree.DEFAULTS,
        "param_schema": {
            "max_depth": {
                "type": "int_nullable",
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Maximum depth of the tree. None = unlimited.",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "step": 1,
                "description": "Minimum samples required to split an internal node.",
            },
            "min_samples_leaf": {
                "type": "int",
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Minimum samples required to be at a leaf node.",
            },
            "criterion": {
                "type": "select",
                "options": ["gini", "entropy"],
                "description": "Function to measure split quality.",
            },
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "class": RandomForest,
        "defaults": RandomForest.DEFAULTS,
        "param_schema": {
            "n_estimators": {
                "type": "int",
                "min": 10,
                "max": 500,
                "step": 10,
                "description": "Number of trees in the forest.",
            },
            "max_depth": {
                "type": "int_nullable",
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Maximum depth of each tree. None = unlimited.",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "step": 1,
                "description": "Minimum samples required to split an internal node.",
            },
            "max_features": {
                "type": "select",
                "options": ["sqrt", "log2"],
                "description": "Number of features to consider at each split.",
            },
            "bootstrap": {
                "type": "bool",
                "description": "Whether to use bootstrap samples when building trees.",
            },
        },
    },
    "knn": {
        "label": "K-Nearest Neighbours",
        "class": KNN,
        "defaults": KNN.DEFAULTS,
        "param_schema": {
            "n_neighbors": {
                "type": "int",
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Number of neighbours to use for prediction.",
            },
            "weights": {
                "type": "select",
                "options": ["uniform", "distance"],
                "description": "Weight function for prediction. distance weights closer neighbours more.",
            },
            "metric": {
                "type": "select",
                "options": ["minkowski", "euclidean", "manhattan"],
                "description": "Distance metric used for finding neighbours.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# train_model()
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_k: int = 5,
) -> dict[str, Any]:
    """
    Train a single classifier and return a comprehensive results dict.

    Parameters
    ----------
    model_name : str
        Key from MODEL_REGISTRY (e.g. 'logistic_regression').
    params : dict
        Hyperparameter overrides. Missing keys fall back to model DEFAULTS.
    X_train, X_test : pd.DataFrame
        Feature splits from the pipeline.
    y_train, y_test : pd.Series
        Target splits from the pipeline.
    cv_k : int
        Number of cross-validation folds (default 5).

    Returns
    -------
    dict with keys:
        model_name, label, params_used, training_time_s,
        accuracy, precision, recall, f1,
        classification_report (str), confusion_matrix (list[list]),
        cv_scores (list[float]), cv_mean (float), cv_std (float),
        class_labels (list)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[model_name]
    ModelClass = entry["class"]
    merged_params = {**entry["defaults"], **params}

    instance = ModelClass(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=merged_params,
    )

    t0 = time.perf_counter()
    instance.train()
    training_time = round(time.perf_counter() - t0, 4)

    preds = instance.predict()
    scores = instance.scores(preds)

    # Cross-validation
    cv_raw = instance.cross_validation(k=cv_k)
    cv_scores = [round(float(s), 4) for s in cv_raw]

    # Confusion matrix as plain list
    cm = confusion_matrix(y_test, preds)
    class_labels = [str(c) for c in instance.model.classes_]

    # ROC-AUC (binary only via predict_proba if available)
    roc_auc = None
    if hasattr(instance, "predict_probs"):
        try:
            probs = instance.predict_probs()
            if probs.shape[1] == 2:
                roc_auc = round(float(roc_auc_score(y_test, probs[:, 1])), 4)
        except Exception:
            pass

    return {
        "model_name": model_name,
        "label": entry["label"],
        "params_used": merged_params,
        "training_time_s": training_time,
        "accuracy": round(scores["accuracy"], 4),
        "precision": round(scores["precision"], 4),
        "recall": round(scores["recall"], 4),
        "f1": round(scores["f1"], 4),
        "roc_auc": roc_auc,
        "classification_report": instance.classification_report(preds),
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "cv_scores": cv_scores,
        "cv_mean": round(float(np.mean(cv_raw)), 4),
        "cv_std": round(float(np.std(cv_raw)), 4),
    }
