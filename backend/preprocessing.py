import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

'''
steps of preprocessing:
Impute
Encode - categorical
Scale - numerical
all this has to be configurable by user - pass dict containing choices
'''

class Preprocessor():

    DEFAULTS = {
        "missing_numeric_strategy": "mean",
        "missing_categorical_strategy": "most_frequent",
        "missing_fill_value": None,
        "scaling": "standard",
        "encoding": "onehot",
        "target_column": None
    }

    def __init__(self, config:dict=None):
        cfg = {**self.DEFAULTS, **(config or {})}

        #dictionary items
        self.missing_numeric_strategy = cfg['missing_numeric_strategy']
        self.missing_categorical_strategy = cfg['missing_categorical_strategy']
        self.missing_fill_value = cfg['missing_fill_value']
        self.encoding = cfg['encoding']
        self.scaling = cfg['scaling']
        self.target_col = cfg['target_column']

        #after fitting
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = StandardScaler
        self.label_encoders = {}
        self.onehot_encoder = None
        self.categorical_cols = []
        self.numeric_cols = []
        self.onehot_feature_names = []

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        target_series = None

        if self.target_col and self.target_col in df.columns:
            target_series = df.pop(self.target_col)
        
        self.identify_cols(df)
        df = self._impute(df, fit=True)
        df = self._encode(df, fit=True)
        df = self._scale(df, fit=True)

        if target_series is not None:
            df[self.target_col] = target_series.values
        
        return df

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        self.check_fitted()
        target_series = None

        if self.target_col and self.target_col in df.columns:
            target_series = df.pop(self.target_col)
        
        df = self._impute(df, fit=False)
        df = self._encode(df, fit=False)
        df = self._scale(df, fit=False)

        if target_series is not None:
            df[self.target_col] = target_series.values

        return df 
    
    def identify_cols(self, df:pd.DataFrame):
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    def _impute(self, df: pd.DataFrame, fit:bool):

        df = df.replace({None: np.nan})

        if self.numeric_cols:
            fill = self.missing_fill_value if self.missing_fill_value is not None else 0
            if fit:
                self.numeric_imputer = SimpleImputer(
                    strategy = self.missing_numeric_strategy,
                    fill_value = fill if self.missing_numeric_strategy == "constant" else None,
                )
                df[self.numeric_cols] = self.numeric_imputer.fit_transform(df[self.numeric_cols])
            else:
                df[self.numeric_cols] = self.numeric_imputer.transform(df[self.numeric_cols])

        if self.categorical_cols:
            fill = self.missing_fill_value if self.missing_fill_value is not None else "missing"
            if fit:
                self.categorical_imputer = SimpleImputer(
                    strategy = self.missing_categorical_strategy,
                    fill_value = fill if self.missing_categorical_strategy == "constant" else None,
                )
                df[self.categorical_cols] = self.categorical_imputer.fit_transform(df[self.categorical_cols])
            else:
                df[self.categorical_cols] = self.categorical_imputer.transform(df[self.categorical_cols])
            
        return df
            
    def _encode(self, df: pd.DataFrame, fit:bool):
        if not self.categorical_cols:
            return df
        
        if self.encoding == "label":
            for col in self.categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    encoded = []
                    for v in df[col].astype(str):
                        if v in le.classes_:
                            encoded.append(le.transform([v])[0])
                        else:
                            encoded.append(-1)
                    
                    df[col] = encoded
        
        elif self.encoding == "onehot":
            if fit:
                self.onehot_encoder = OneHotEncoder(
                    sparse_output = False, handle_unknown = "ignore", drop = None
                )
                encoded = self.onehot_encoder.fit_transform(df[self.categorical_cols])
                self.onehot_feature_names = list(
                    self.onehot_encoder.get_feature_names_out(self.categorical_cols)
                )
            else:
                encoded = self.onehot_encoder.transform(df[self.categorical_cols])
            
            encoded_df = pd.DataFrame(encoded, columns = self.onehot_feature_names, index = df.index)
            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis = 1)
        
        return df
    
    def _scale(self, df:pd.DataFrame, fit:bool):
        if not self.scaling or not self.numeric_cols:
            return df
        
        if self.scaling == "standard":
            scaler = StandardScaler
        elif self.scaling == "minmax":
            scaler = MinMaxScaler
        else:
            raise ValueError(f"unknown strategy selected")
        
        if fit:
            self.scaler = scaler()
            df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        else:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df
    
    def check_fitted(self):
        if self.numeric_imputer is None and self.categorical_imputer is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform() first.")
