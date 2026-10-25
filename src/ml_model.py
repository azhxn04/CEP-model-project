"""
Machine Learning Component Module
Trains, evaluates, and compares structured tabular models:
- Logistic Regression (Baseline 1)
- Random Forest (Baseline 2)
- XGBoost Classifier (Primary ML Model)

CRITICAL ML BEST PRACTICE (Label Leakage Prevention):
Directly predicting 'Item' using synthetic 'Customer Requirement' creates artificial 100% shortcut leakage.
This module trains behavioral models using true demographic, temporal, and interaction features:
[Age, Gender, Purchase Mode, Time Spent (Minutes), Shift, Price Sensitivity, Discount Rate, Quantity].
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class CustomerMLModel:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.evaluation_results = {}
        self.feature_names = []
        self.classes_ = []

    def prepare_data(self, history_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extracts clean non-leaked tabular features and target labels.
        Target: Predict product category interest based on customer profile & session behavior.
        """
        df = history_df.copy()

        # Features
        feature_cols = ["Age", "Gender", "Purchase Mode", "Time Spent (Minutes)", "Shift", "Discount Rate (%)", "Quantity"]
        target_col = "Item"

        # Ensure no nulls
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        return X, y

    def train_and_evaluate(self, history_df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains Logistic Regression, Random Forest, and XGBoost on a held-out test split.
        Returns comprehensive evaluation metrics and saves artifacts.
        """
        X, y_raw = self.prepare_data(history_df)

        # Encode target classes
        y = self.label_encoder.fit_transform(y_raw)
        self.classes_ = list(self.label_encoder.classes_)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Preprocessing Pipeline
        numeric_features = ["Age", "Time Spent (Minutes)", "Discount Rate (%)", "Quantity"]
        categorical_features = ["Gender", "Purchase Mode", "Shift"]

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Fit preprocessor on X_train only
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Reconstruct feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))
        self.feature_names = numeric_features + cat_cols

        # Model Definitions
        model_candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state),
            "XGBoost": XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                random_state=random_state,
                eval_metric="mlogloss"
            )
        }

        results = {}

        for name, clf in model_candidates.items():
            clf.fit(X_train_proc, y_train)
            self.models[name] = clf

            # Predictions
            y_pred = clf.predict(X_test_proc)
            y_prob = clf.predict_proba(X_test_proc) if hasattr(clf, "predict_proba") else None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()

            # Multiclass ROC-AUC
            roc_auc = None
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                except Exception:
                    roc_auc = 0.50

            # Feature Importance
            feature_imp = {}
            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                feature_imp = {fn: float(val) for fn, val in zip(self.feature_names, imp)}
            elif hasattr(clf, "coef_"):
                imp = np.mean(np.abs(clf.coef_), axis=0)
                feature_imp = {fn: float(val) for fn, val in zip(self.feature_names, imp)}

            results[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "confusion_matrix": cm,
                "feature_importance": feature_imp,
                "classification_report": classification_report(y_test, y_pred, target_names=self.classes_, output_dict=True, zero_division=0)
            }

        self.evaluation_results = {
            "metrics": results,
            "classes": self.classes_,
            "feature_names": self.feature_names,
            "test_size": test_size,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        # Save artifacts
        self.save_artifacts()
        return self.evaluation_results

    def predict_category_affinity(self, customer_features: Dict[str, Any], model_name: str = "XGBoost") -> Dict[str, float]:
        """
        Predicts category probability distribution for a given customer profile.
        Used as the Behavioral ML signal in hybrid recommendation ranking.
        """
        if model_name not in self.models or self.preprocessor is None:
            self.load_artifacts()

        clf = self.models.get(model_name)
        if clf is None or self.preprocessor is None:
            return {c: 1.0 / len(self.classes_) for c in self.classes_}

        df_input = pd.DataFrame([customer_features])
        # Default missing values if needed
        default_vals = {"Age": 32, "Gender": "Male", "Purchase Mode": "Online", "Time Spent (Minutes)": 45, "Shift": "Evening", "Discount Rate (%)": 15.0, "Quantity": 1}
        for k, v in default_vals.items():
            if k not in df_input.columns or df_input[k].isnull().any():
                df_input[k] = v

        proc_input = self.preprocessor.transform(df_input)
        probs = clf.predict_proba(proc_input)[0]

        return {cat: float(prob) for cat, prob in zip(self.classes_, probs)}

    def save_artifacts(self):
        """Saves models, preprocessor, and evaluation metrics."""
        joblib.dump(self.models, os.path.join(self.models_dir, "trained_ml_models.joblib"))
        joblib.dump(self.preprocessor, os.path.join(self.models_dir, "preprocessor.joblib"))
        joblib.dump(self.label_encoder, os.path.join(self.models_dir, "label_encoder.joblib"))
        joblib.dump(self.evaluation_results, os.path.join(self.models_dir, "ml_evaluation_results.joblib"))
        print(f"[CustomerMLModel] Saved ML models and evaluation to {self.models_dir}")

    def load_artifacts(self):
        """Loads models and preprocessor from disk."""
        models_path = os.path.join(self.models_dir, "trained_ml_models.joblib")
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        le_path = os.path.join(self.models_dir, "label_encoder.joblib")
        eval_path = os.path.join(self.models_dir, "ml_evaluation_results.joblib")

        if os.path.exists(models_path) and os.path.exists(prep_path):
            self.models = joblib.load(models_path)
            self.preprocessor = joblib.load(prep_path)
            self.label_encoder = joblib.load(le_path)
            self.classes_ = list(self.label_encoder.classes_)
            if os.path.exists(eval_path):
                self.evaluation_results = joblib.load(eval_path)
            return True
        return False
