import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class RandomForest:

    DEFAULTS = {
        'max_depth': None,
        'n_estimators': 100,
        'min_samples_split': 2,
        'max_features': 'sqrt',
        'bootstrap': True
    }

    def __init__(self, X_train, X_test, y_train, y_test, config:dict = None):
        self.cfg = {**self.DEFAULTS, **(config or {})}
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators = self.cfg['n_estimators'],
                                            max_depth = self.cfg['max_depth'],
                                            min_samples_split = self.cfg['min_samples_split'],
                                            max_features = self.cfg['max_features'],
                                            bootstrap = self.cfg['bootstrap'])

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

    def classification_report(self, preds):
        return classification_report(self.y_test, preds)

    def confusion_matrix(self, preds):
        conf = confusion_matrix(self.y_test, preds)
        disp = ConfusionMatrixDisplay(conf,
                                      display_labels = self.model.classes_)
        disp.plot()
        plt.title(f"{self.y_test.name} Confusion Matrix")
        plt.show()


