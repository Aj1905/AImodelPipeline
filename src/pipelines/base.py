"""
機械学習パイプラインの抽象基底クラス。

すべてのパイプライン実装が従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from ..features import FeatureManager
from ..models.base import BaseModel
from ..evaluation.results import CrossValidationResult, TrainingResult
from ..features.managers.target_manager import TargetManager


class BasePipeline(ABC):
    """機械学習パイプラインの抽象基底クラス。"""

    def __init__(
        self,
        model: BaseModel,
        target_manager: TargetManager,
        feature_manager: FeatureManager,
    ):
        """BasePipelineを初期化する。

        Args:
            model (BaseModel): 使用するモデル
            target_manager (TargetManager): ターゲット管理クラス.
            feature_manager (FeatureManager): 特徴量管理クラス.
        """
        self._model = model
        self._feature_manager = feature_manager
        self._target_manager = target_manager

    def load_data(self, data: pl.DataFrame, target_column: str, feature_names: list[str] | None = None) -> None:
        """データをFeatureManagerとTargetManagerに読み込む。

        Args:
            data (pl.DataFrame): 学習データ
            target_column (str): ターゲット列名
            feature_names (list[str] | None, optional): 使用する特徴量名のリスト. Noneの場合はターゲット列以外のすべての列を使用.

        Raises:
            ValueError: ターゲット列が存在しない場合
        """
        if target_column not in data.columns:
            raise ValueError(f"ターゲット列 '{target_column}' がデータに存在しません")

        self._target_manager.update_from_dataframe(data, target_column)

        feature_data = data.drop(target_column)
        self._feature_manager.update_features_from_dataframe(feature_data, feature_names)

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
            NotImplementedError: サブクラスで実装する必要がある
            ValueError: データが読み込まれていない場合
        """
        pass

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
            NotImplementedError: サブクラスで実装する必要がある
            ValueError: データが読み込まれていない場合
        """
        pass

    def save_model(self, file_path: str | Path) -> None:
        """学習済みモデルを保存する。

        Args:
            file_path (str | Path): 保存先ファイルパス

        Raises:
            RuntimeError: モデルが学習されていない場合
        """
        if not self._model.is_trained():
            raise RuntimeError("モデルが学習されていません")

        self._model.save_model(file_path)

    def load_model(self, file_path: str | Path) -> None:
        """学習済みモデルを読み込む。

        Args:
            file_path (str | Path): 読み込み元ファイルパス
        """
        self._model.load_model(file_path)

    def get_model(self) -> BaseModel:
        """モデルを取得する。

        Returns:
            BaseModel: モデル
        """
        return self._model

    def get_feature_manager(self) -> FeatureManager:
        """特徴量管理クラスを取得する。

        Returns:
            FeatureManager: 特徴量管理クラス
        """
        return self._feature_manager

    def get_target_manager(self) -> TargetManager:
        """ターゲット管理クラスを取得する。

        Returns:
            TargetManager: ターゲット管理クラス
        """
        return self._target_manager

    def __repr__(self) -> str:
        """オブジェクトの文字列表現を返す。

        Returns:
            str: オブジェクトの文字列表現
        """
        model_status = "trained" if self._model.is_trained() else "untrained"
        feature_count = self._feature_manager.get_enabled_feature_count()
        return f"{self.__class__.__name__}(model={self._model.get_model_name()}, status={model_status}, features={feature_count})"
