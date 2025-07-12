# octopy/selector.py

from typing import List
import pandas as pd
import numpy as np

class ModelSelector:
    def __init__(self, df: pd.DataFrame, target: str, problem_type: str = None):
        """
        Initialize with dataset, target column, and optional problem type.
        problem_type: 'classification' or 'regression'.
        If None, it tries to infer.
        """
        self.df = df
        self.target = target
        self.problem_type = problem_type or self._infer_problem_type()
        self.num_samples = len(df)
        self.num_features = df.shape[1] - 1  # excluding target
        self.target_unique = df[target].nunique()
        self.imbalance_ratio = self._calculate_imbalance_ratio()

    def _infer_problem_type(self) -> str:
        """
        Infer problem type based on target variable data type and unique values.
        """
        if pd.api.types.is_numeric_dtype(self.df[self.target]):
            if self.df[self.target].nunique() <= 20:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def _calculate_imbalance_ratio(self) -> float:
        """
        Calculate imbalance ratio for classification problems.
        Ratio = count_majority_class / count_minority_class
        Returns 1 for regression problems.
        """
        if self.problem_type != 'classification':
            return 1.0

        counts = self.df[self.target].value_counts()
        if len(counts) == 0:
            return 1.0
        return counts.max() / counts.min()

    def suggest_models(self) -> List[str]:
        """
        Suggest models based on problem type, dataset size, feature count, and imbalance.
        """
        if self.problem_type == 'classification':
            return self._suggest_classification_models()
        elif self.problem_type == 'regression':
            return self._suggest_regression_models()
        else:
            return []

    def _suggest_classification_models(self) -> List[str]:
        models = []

        # Small datasets (< 1000 samples)
        if self.num_samples < 1000:
            models += ["Logistic Regression", "K-Nearest Neighbors"]
        else:
            models += ["Random Forest Classifier", "XGBoost Classifier", "Gradient Boosting Classifier"]

        # High-dimensional data (> 50 features)
        if self.num_features > 50:
            models.append("Support Vector Machine (with kernel)")

        # Imbalanced data
        if self.imbalance_ratio > 3:
            models.append("Balanced Random Forest")
            models.append("SMOTE + Any Classifier")

        # Simple baseline
        models.append("Dummy Classifier (baseline)")

        # Remove duplicates
        return list(dict.fromkeys(models))

    def _suggest_regression_models(self) -> List[str]:
        models = []

        # Small datasets
        if self.num_samples < 1000:
            models += ["Linear Regression", "K-Nearest Neighbors Regressor"]
        else:
            models += ["Random Forest Regressor", "XGBoost Regressor", "Gradient Boosting Regressor"]

        # High-dimensional data
        if self.num_features > 50:
            models.append("Support Vector Regressor (with kernel)")

        # Simple baseline
        models.append("Dummy Regressor (baseline)")

        return list(dict.fromkeys(models))

    def print_summary(self):
        """
        Print detailed summary of data and recommended models.
        """
        print(f"Problem type: {self.problem_type}")
        print(f"Samples: {self.num_samples}")
        print(f"Features (excluding target): {self.num_features}")
        if self.problem_type == 'classification':
            print(f"Target classes: {self.target_unique}")
            print(f"Class imbalance ratio: {self.imbalance_ratio:.2f}")
        print("\nRecommended models:")
        for m in self.suggest_models():
            print(f"- {m}")
