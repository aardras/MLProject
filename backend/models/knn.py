import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

class KNN:

    DEFAULTS = {
        'n_neighbors': 5,
        'weights':'uniform',
        'metric':'minkowski'
    }

    def __init__(self, X_train, X_test, y_train, y_test, config:dict = None):
        self.cfg = {**self.DEFAULTS, **(config or {})}
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = KNeighborsClassifier(n_neighbors = self.cfg['n_neighbors'],
                         weights = self.cfg['weights'],
                         metric = self.cfg['metric'])

    def train(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def scores(self, preds):
        return {
            "accuracy": accuracy_score(self.y_test, preds),
            "precision": precision_score(self.y_test, preds),
            "recall": recall_score(self.y_test, preds),
            "f1": f1_score(self.y_test, preds),
        }
    
    def classification_report(self, preds):
        return classification_report(self.y_test, preds)
    
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


