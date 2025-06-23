"""
特徴量管理のインターフェース定義。

特徴量管理クラスが従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod

import polars as pl


class IFeatureManager(ABC):
    """特徴量管理のインターフェース。"""

    @abstractmethod
    def add_features(self, df: pl.DataFrame, enabled: bool = True) -> None:
        """特徴量を追加する。

        Args:
            df (pl.DataFrame): 特徴量を含むDataFrame
            enabled (bool, optional): 特徴量を有効にするかどうか
        """
        pass

    @abstractmethod
    def remove_feature(self, name: str) -> None:
        """特徴量を削除する。

        Args:
            name (str): 削除する特徴量の名称
        """
        pass

    @abstractmethod
    def enable_feature(self, name: str) -> None:
        """特徴量を有効にする。

        Args:
            name (str): 有効にする特徴量の名称
        """
        pass

    @abstractmethod
    def disable_feature(self, name: str) -> None:
        """特徴量を無効にする。

        Args:
            name (str): 無効にする特徴量の名称
        """
        pass

    @abstractmethod
    def get_enabled_features(self) -> pl.DataFrame:
        """有効な特徴量のみをDataFrameとして取得する。

        Returns:
            pl.DataFrame: 有効な特徴量を含むDataFrame
        """
        pass

    @abstractmethod
    def get_all_features(self) -> pl.DataFrame:
        """すべての特徴量をDataFrameとして取得する。

        Returns:
            pl.DataFrame: すべての特徴量を含むDataFrame
        """
        pass

    @abstractmethod
    def get_feature_count(self) -> int:
        """特徴量の総数を取得する。

        Returns:
            int: 特徴量の総数
        """
        pass

    @abstractmethod
    def get_enabled_feature_count(self) -> int:
        """有効な特徴量の数を取得する。

        Returns:
            int: 有効な特徴量の数
        """
        pass
