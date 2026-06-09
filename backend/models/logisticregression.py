import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class LogisticRegressionModel:

    DEFAULTS = {
        'C': 1.0,
        'max_iter':100,
        'solver': 'lbfgs',
        'penalty': 'l2'
    }

    def __init__(self, X_train, X_test, y_train, y_test, config:dict = None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cfg = {**self.DEFAULTS, **(config or {})}
        self.model = LogisticRegression(penalty=self.cfg['penalty'],
                                        C = self.cfg['C'],
                                        random_state=42,
                                        solver = self.cfg['solver'],
                                        max_iter=self.cfg['max_iter'])

    def train(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def scores(self, preds):
        return {
            "accuracy": accuracy_score(self.y_test, preds),
            "precision": precision_score(self.y_test, preds),
            "recall": recall_score(self.y_test, preds),
            "f1": f1_score(self.y_test, preds)
        }
    
    def cross_validation(self, k=5):
        X_df = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_df = pd.concat([self.y_train, self.y_test], ignore_index = True)
        return cross_val_score(
            self.model,
            X_df,
            y_df,
            cv=k
        )
    def predict_probs(self):
        return self.model.predict_proba(self.X_test)
    
    def get_params(self):
        return self.cfg

    def confusion_matrix(self, preds):
        conf = confusion_matrix(self.y_test, preds)
        disp = ConfusionMatrixDisplay(conf,
                                      display_labels = self.model.classes_)
        disp.plot()
        plt.title(f"{self.y_test.name} Confusion Matrix")
        plt.show()

    def classification_report(self, preds):
        return classification_report(self.y_test, preds)
