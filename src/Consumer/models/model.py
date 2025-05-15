from typing import List
import warnings
import numpy as np
import pickle
import os

from joblibspark import register_spark
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import parallel_backend
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from pyspark.sql.dataframe import DataFrame

from models.feature_extractor import *

warnings.filterwarnings('ignore')
register_spark()

class RandomForest:
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 random_state=42, 
                 feature_extractor=HOGFeatureExtractor()
    ):
        self.feature_extractor = feature_extractor
        self.model = RandomForestClassifier(n_estimators=n_estimators, 
                                            max_depth=max_depth, 
                                            random_state=random_state)

    def train(self, df: DataFrame, save_path: str = None) -> List:
        # Collect to driver
        X_raw = np.array(df.select("image").collect())
        y = np.array(df.select("label").collect()).reshape(-1)
        # print(X_raw)

        features = []
        for img in X_raw:
            img = img.reshape(32, 32, 3)
            feat = self.feature_extractor.extract(img)
            features.append(feat)

        X = np.array(features)

        # Train with Spark backend
        with parallel_backend("spark", n_jobs=1):
            self.model.fit(X, y)

        # Save model:
        if save_path:
            self.save(save_path)

        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")

        return y_pred, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        X_raw = np.array(df.select("image").collect())
        y = np.array(df.select("label").collect()).reshape(-1)

        features = []
        for img in X_raw:
            img = img.reshape(32, 32, 3)
            feat = self.feature_extractor.extract(img)
            features.append(feat)

        X = np.array(features)
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")

        return y, y_pred, accuracy, precision, recall, f1
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_extractor = data['feature_extractor']
        print(f"Model loaded from {path}")

class SVM:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: str = 'scale',
        random_state: int = 42,
        feature_extractor: HOGFeatureExtractor = HOGFeatureExtractor()
    ):
        self.feature_extractor = feature_extractor
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            probability=True,   
            random_state=random_state
        )

    def train(self, df: DataFrame, save_path: str = None) -> List:
        # Collect to driver
        X_raw = np.array(df.select("image").collect())
        y = np.array(df.select("label").collect()).reshape(-1)

        features = []
        for img in X_raw:
            img = img.reshape(32, 32, 3)
            feat = self.feature_extractor.extract(img)
            features.append(feat)

        X = np.array(features)

        # Train with Spark backend
        with parallel_backend("spark", n_jobs=1):
            self.model.fit(X, y)

        # Save model:
        if save_path:
            self.save(save_path)

        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")

        return y_pred, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        X_raw = np.array(df.select("image").collect())
        y = np.array(df.select("label").collect()).reshape(-1)

        features = []
        for img in X_raw:
            img = img.reshape(32, 32, 3)
            feat = self.feature_extractor.extract(img)
            features.append(feat)

        X = np.array(features)
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")

        return y, y_pred, accuracy, precision, recall, f1
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_extractor = data['feature_extractor']
        print(f"Model loaded from {path}")