from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .feature_manager import FeatureManager
from .target_manager import TargetManager


class TreeModelPipeline:
    """Simple pipeline for tree-based models."""

    def __init__(self, model: Any, feature_manager: FeatureManager, target_manager: TargetManager) -> None:
        self.model = model
        self.feature_manager = feature_manager
        self.target_manager = target_manager

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate various evaluation metrics."""
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def _check_overfitting(self, train_metrics: dict[str, float], test_metrics: dict[str, float]) -> dict[str, Any]:
        """Check for overfitting by comparing train and test metrics."""
        overfitting_indicators = {}

        # R²の差を計算
        r2_diff = train_metrics["r2"] - test_metrics["r2"]
        overfitting_indicators["r2_difference"] = r2_diff

        # RMSEの差を計算(テストの方が高い場合が過学習の可能性)
        rmse_diff = test_metrics["rmse"] - train_metrics["rmse"]
        overfitting_indicators["rmse_difference"] = rmse_diff

        # 過学習の判断
        is_overfitting = False
        overfitting_reasons = []

        if r2_diff > 0.1:  # R²の差が10%以上
            is_overfitting = True
            overfitting_reasons.append(f"R²の差が大きい ({r2_diff:.4f})")

        if rmse_diff > 0:  # テストRMSEが学習RMSEより高い
            overfitting_reasons.append(f"テストRMSEが学習RMSEより高い ({rmse_diff:.4f})")

        if test_metrics["r2"] < 0.5 and train_metrics["r2"] > 0.7:
            is_overfitting = True
            overfitting_reasons.append("学習データでは高い性能だが、テストデータでは低性能")

        overfitting_indicators["is_overfitting"] = is_overfitting
        overfitting_indicators["reasons"] = overfitting_reasons

        return overfitting_indicators

    def train(self, test_size: float = 0.2, random_state: int = 42) -> dict[str, Any]:
        """Train the model and return comprehensive evaluation metrics."""
        if self.feature_manager.features is None or self.target_manager.target is None:
            raise ValueError("Features or target not set")

        x = self.feature_manager.features
        y = self.target_manager.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # モデルを学習
        self.model.fit(x_train, y_train)

        # 予測を実行
        y_train_pred = self.model.predict(x_train)
        y_test_pred = self.model.predict(x_test)

        # 評価指標を計算
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)

        # 過学習のチェック
        overfitting_analysis = self._check_overfitting(train_metrics, test_metrics)

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "overfitting_analysis": overfitting_analysis,
        }
