"""
機械学習モデルの抽象基底クラス。

すべてのモデル実装が従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl


class BaseModel(ABC):
    """機械学習モデルの抽象基底クラス。"""

    def __init__(self, model_name: str | None = None):
        """BaseModelを初期化する。

        Args:
            model_name (str | None, optional): モデルの名称. デフォルトは None.
        """
        self.model_name = model_name or self.__class__.__name__
        self._model: Any = None
        self._is_trained: bool = False

    @abstractmethod
    def train(self, features: pl.DataFrame, targets: pl.Series) -> None:
        """モデルを学習する。

        Args:
            features (pl.DataFrame): 特徴量データ
            targets (pl.Series): 正解データ

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        pass

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> pl.Series:
        """予測を実行する。

        Args:
            features (pl.DataFrame): 特徴量データ

        Returns:
            pl.Series: 予測結果

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        pass

    @abstractmethod
    def save_model(self, file_path: str | Path) -> None:
        """学習済みモデルを保存する。

        Args:
            file_path (str | Path): 保存先ファイルパス

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str | Path) -> None:
        """学習済みモデルを読み込む。

        Args:
            file_path (str | Path): 読み込み元ファイルパス

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        pass

    def is_trained(self) -> bool:
        """モデルが学習済みかどうかを確認する。

        Returns:
            bool: 学習済みの場合は True、そうでない場合は False
        """
        return self._is_trained

    def get_model_name(self) -> str:
        """モデル名を取得する。

        Returns:
            str: モデル名
        """
        return self.model_name

    def set_model_name(self, name: str) -> None:
        """モデル名を設定する。

        Args:
            name (str): 設定するモデル名
        """
        self.model_name = name

    def __repr__(self) -> str:
        """オブジェクトの文字列表現を返す。

        Returns:
            str: オブジェクトの文字列表現
        """
        status = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')"
