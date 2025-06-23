"""
機械学習モデルのインターフェース定義。

すべてのモデル実装が従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl


class IModel(ABC):
    """機械学習モデルのインターフェース。"""

    @abstractmethod
    def train(self, features: pl.DataFrame, targets: pl.Series) -> None:
        """モデルを学習する。

        Args:
            features (pl.DataFrame): 特徴量データ
            targets (pl.Series): 正解データ
        """
        pass

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> pl.Series:
        """予測を実行する。

        Args:
            features (pl.DataFrame): 特徴量データ

        Returns:
            pl.Series: 予測結果
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
    def get_feature_importance(self) -> dict[str, float]:
        """特徴量重要度を取得する。

        Returns:
            dict[str, float]: 特徴量名と重要度の辞書
        """
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """モデルが学習済みかどうかを確認する。

        Returns:
            bool: 学習済みの場合は True、そうでない場合は False
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """モデル名を取得する。

        Returns:
            str: モデル名
        """
        pass 