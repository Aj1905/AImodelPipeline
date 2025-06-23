"""
機械学習パイプライン。

特徴量、モデル、データを統合して学習と評価を行うパイプラインを提供します。
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .features import FeatureManager
from .models.base import BaseModel
from .results import CrossValidationResult, TrainingResult
from .targets import TargetManager


class TrainingPipeline(ABC):
    """機械学習の学習パイプライン。"""

    @abstractmethod
    def train(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingResult:
        """事前に読み込まれたデータを使ってモデルを学習する。

        Args:
            test_size (float, optional): テストデータの割合. デフォルトは 0.2.
            random_state (int, optional): ランダムシード. デフォルトは 42.

        Returns:
            TrainingResult: 学習結果

        Raises:
            ValueError: データが読み込まれていない場合
        """

    @abstractmethod
    def cross_validate(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> CrossValidationResult:
        """事前に読み込まれたデータでクロスバリデーションを実行する。

        Args:
            cv_folds (int, optional): クロスバリデーションの分割数. デフォルトは 5.
            random_state (int, optional): ランダムシード. デフォルトは 42.

        Returns:
            CrossValidationResult: クロスバリデーション結果

        Raises:
            ValueError: データが読み込まれていない場合
        """

    def __init__(
        self,
        model: BaseModel,
        feature_manager: FeatureManager,
        target_manager: TargetManager
    ):
        """TrainingPipelineを初期化する。

        Args:
            model (BaseModel): 使用するモデル
            feature_manager (FeatureManager): 特徴量管理クラス
            target_manager (TargetManager): ターゲット管理クラス.
        """
        feature_len = len(feature_manager)
        target_len = len(target_manager)
        if feature_len == 0:
            raise ValueError("FeatureManagerに特徴量が設定されていません")
        if target_len == 0:
            raise ValueError("TargetManagerにターゲットが設定されていません")
        if feature_len != target_len:
            raise ValueError(
                f"FeatureManagerとTargetManagerの行数が一致しません "
                f"[features][{feature_len}][target][{target_len}]"
            )
        self.model = model
        self.feature_manager = feature_manager
        self.target_manager = target_manager
        self._train_metrics: dict[str, float] | None = None
        self._val_metrics: dict[str, float] | None = None

    def save_model(self, file_path: str | Path) -> None:
        """学習済みモデルを保存する。

        Args:
            file_path (str | Path): 保存先ファイルパス

        Raises:
            RuntimeError: モデルが学習されていない場合
        """
        if not self.model.is_trained():
            raise RuntimeError("モデルが学習されていません")

        self.model.save_model(file_path)

    def get_train_metrics(self) -> dict[str, float] | None:
        """学習データの評価指標を取得する。

        Returns:
            dict[str, float] | None: 学習データの評価指標
        """
        return self._train_metrics

    def get_validation_metrics(self) -> dict[str, float] | None:
        """検証データの評価指標を取得する。

        Returns:
            dict[str, float] | None: 検証データの評価指標
        """
        return self._val_metrics

    def __repr__(self) -> str:
        """オブジェクトの文字列表現を返す。

        Returns:
            str: オブジェクトの文字列表現
        """
        model_status = "trained" if self.model.is_trained() else "untrained"
        feature_count = self.feature_manager.get_enabled_feature_count()
        return (
            f"TrainingPipeline(model={self.model.get_model_name()}, "
            f"status={model_status}, features={feature_count})"
        )
