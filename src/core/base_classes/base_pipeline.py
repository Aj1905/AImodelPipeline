"""
機械学習パイプラインの基底クラス。

すべてのパイプライン実装の基底となるクラスを定義します。
"""

from pathlib import Path

import polars as pl

from ...features.managers.feature_manager import FeatureManager
from ...features.managers.target_manager import TargetManager
from ..interfaces.model_interface import IModel
from ..interfaces.pipeline_interface import IPipeline


class BasePipeline(IPipeline):
    """機械学習パイプラインの基底クラス。"""

    def __init__(
        self,
        model: IModel,
        target_manager: TargetManager,
        feature_manager: FeatureManager,
    ):
        """BasePipelineを初期化する。

        Args:
            model (IModel): 使用するモデル
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

    def get_model(self) -> IModel:
        """モデルを取得する。

        Returns:
            IModel: モデル
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
