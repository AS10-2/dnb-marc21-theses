import joblib
import logging
import pandas as pd

from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.cleaning import clean_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetroClassifier:
    """
    Leakage-sichere ML-Pipeline für Retro-Klassifikation.
    """

    def __init__(self):
        self.model: Optional[Pipeline] = None

    @staticmethod
    def build_pipeline() -> Pipeline:
        """
        Erzeugt TF-IDF + LogisticRegression Pipeline.
        """
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text,
                max_features=200_000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

    def evaluate(self, model: Pipeline, X_test, y_test):
        """
        Evaluiert Modell und gibt Report zurück.
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info("\n%s", report)
        return report

    def cross_validate(self, df: pd.DataFrame, cv: int = 5):
        """
        Optionale Cross-Validation zur Stabilitätsprüfung.
        """
        X = df["text"]
        y = df["label"]

        pipeline = self.build_pipeline()
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")

        logger.info("CV F1-Scores: %s", scores)
        logger.info("Mean F1: %.4f", scores.mean())

        return scores

    def train(self, df: pd.DataFrame, retrain_full: bool = True):
        """
        Training auf alle ausgewählten Daten.  
        """

        X, y = df["text"], df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)
        self.evaluate(pipeline, X_test, y_test)

        if retrain_full:
            # Finales Modell auf allen Daten
            pipeline.fit(X, y)
            logger.info("Retrained on full dataset for inference.")

        self.model = pipeline
        return pipeline
        
    def predict(self, texts: pd.Series) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        return pd.DataFrame({
            "retro_prob":  self.model.predict_proba(texts)[:, 1],
            "retro_label": self.model.predict(texts),
        }, index=texts.index)

    def save(self, path: str = "models/retro_classifier.joblib"):
        """
        Speichert trainiertes Modell.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def load(path: str):
        """
        Lädt gespeichertes Modell.
        """
        return joblib.load(path)
