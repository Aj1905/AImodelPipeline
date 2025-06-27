from __future__ import annotations

from typing import Any

from sklearn.model_selection import train_test_split

from .feature_manager import FeatureManager
from .target_manager import TargetManager


class TreeModelPipeline:
    """Simple pipeline for tree-based models."""

    def __init__(self, model: Any, feature_manager: FeatureManager, target_manager: TargetManager) -> None:
        self.model = model
        self.feature_manager = feature_manager
        self.target_manager = target_manager

    def train(self, test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
        """Train the model and return R² scores."""
        if self.feature_manager.features is None or self.target_manager.target is None:
            raise ValueError("Features or target not set")
        x = self.feature_manager.features
        y = self.target_manager.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.model.fit(x_train, y_train)
        train_r2 = self.model.score(x_train, y_train)
        test_r2 = self.model.score(x_test, y_test)
        return {"train_r2": float(train_r2), "test_r2": float(test_r2)}
