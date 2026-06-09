import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def shapes(self):
        return {
            "X_train": self.X_train.shape,
            "X_test": self.X_test.shape,
            "y_train": self.y_train.shape,
            "y_test": self.y_test.shape,           
        }
    
class SplitClass:

    DEFAULTS = {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": False,
        "target_column": None,
    }

    def __init__(self, config: dict):
        cfg = {**self.DEFAULTS, **(config or {})}
        self.test_size = cfg["test_size"]
        self.random_state = cfg["random_state"]
        self.stratify = cfg["stratify"]
        self.target_column = cfg["target_column"]
    
    def split(self, df: pd.DataFrame, target_column=None):
        col = target_column or self.target_column
        if col is None:
            raise ValueError("target_column must be set in config")
        
        X = df.drop(columns=[col])
        y = df[col]

        stratify_arg = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )

        return Split(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )       