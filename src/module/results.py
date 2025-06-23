"""
機械学習結果のデータクラス。

学習結果や評価結果を構造化して管理します。
"""

from dataclasses import dataclass

from .metrics import Metrics
from .models.base import BaseModel


@dataclass
class TrainingResult:
    """学習結果を格納するデータクラス。"""

    model: BaseModel
    train_metrics: Metrics
    validation_metrics: Metrics
    train_size: int
    test_size: int
    feature_count: int

    def __str__(self) -> str:
        """学習結果の包括的な文字列表現を返す。"""
        return (
            "📊 学習結果:\n"
            f"訓練データサイズ: {self.train_size}\n"
            f"テストデータサイズ: {self.test_size}\n"
            f"特徴量数: {self.feature_count}\n\n"
            "📈 評価指標:\n"
            f"訓練 - {self.train_metrics}\n"
            f"検証 - {self.validation_metrics}"
        )


@dataclass
class CrossValidationResult:
    """クロスバリデーション結果を格納するデータクラス。"""

    fold_metrics: list[Metrics]
    mean_metrics: Metrics
    std_metrics: Metrics
    cv_folds: int

    def __str__(self) -> str:
        """クロスバリデーション結果の文字列表現を返す。"""
        fold_results = "\n".join([f"  Fold {i + 1}: {metrics}" for i, metrics in enumerate(self.fold_metrics)])
        return (
            f"🔄 クロスバリデーション結果 ({self.cv_folds}フォールド):\n"
            f"{fold_results}\n\n"
            f"📊 平均: {self.mean_metrics}\n"
            f"📊 標準偏差: {self.std_metrics}"
        )


@dataclass
class MetricsResult:
    """評価指標結果を格納するデータクラス。"""

    mse: float
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        """評価指標結果の文字列表現を返す。"""
        return f"MSE: {self.mse:.4f}, RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}, R²: {self.r2:.4f}"
