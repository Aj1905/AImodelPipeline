"""
機械学習パイプラインのインターフェース定義。

すべてのパイプライン実装が従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl

from ..interfaces.model_interface import IModel


class IPipeline(ABC):
    """機械学習パイプラインのインターフェース。"""

    @abstractmethod
    def load_data(self, data: pl.DataFrame, target_column: str, feature_names: list[str] | None = None) -> None:
        """データを読み込む。

        Args:
            data (pl.DataFrame): 学習データ
            target_column (str): ターゲット列名
            feature_names (list[str] | None, optional): 使用する特徴量名のリスト
        """
        pass

    @abstractmethod
    def train(self, test_size: float = 0.2, random_state: int = 42) -> Any:
        """モデルを学習する。

        Args:
            test_size (float, optional): テストデータの割合
            random_state (int, optional): ランダムシード

        Returns:
            Any: 学習結果
        """
        pass

    @abstractmethod
    def cross_validate(self, cv_folds: int = 5, random_state: int = 42) -> Any:
        """クロスバリデーションを実行する。

        Args:
            cv_folds (int, optional): クロスバリデーションの分割数
            random_state (int, optional): ランダムシード

        Returns:
            Any: クロスバリデーション結果
        """
        pass

    @abstractmethod
    def save_model(self, file_path: str | Path) -> None:
        """学習済みモデルを保存する。

        Args:
            file_path (str | Path): 保存先ファイルパス
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str | Path) -> None:
        """学習済みモデルを読み込む。

        Args:
            file_path (str | Path): 読み込み元ファイルパス
        """
        pass

    @abstractmethod
    def get_model(self) -> IModel:
        """モデルを取得する。

        Returns:
            IModel: モデル
        """
        pass
