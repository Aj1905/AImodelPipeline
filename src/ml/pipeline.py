from __future__ import annotations

from typing import Any

from sklearn.model_selection import train_test_split

from .managers import FeatureManager, TargetManager


class TreeModelPipeline:
    """シンプルな学習用パイプライン"""

    def __init__(self, model: Any, feature_manager: FeatureManager, target_manager: TargetManager) -> None:
        self.model = model
        self.feature_manager = feature_manager
        self.target_manager = target_manager

    def train(self, test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
        if self.feature_manager.features is None or self.target_manager.target is None:
            raise ValueError("特徴量またはターゲットが設定されていません")
        X = self.feature_manager.features
        y = self.target_manager.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        train_r2 = self.model.score(X_train, y_train)
        test_r2 = self.model.score(X_test, y_test)
        return {"train_r2": float(train_r2), "test_r2": float(test_r2)}
