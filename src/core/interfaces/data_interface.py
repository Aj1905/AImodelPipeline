"""
データアクセスのインターフェース定義。

データアクセスクラスが従うべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class IDataRepository(ABC):
    """データリポジトリのインターフェース。"""

    @abstractmethod
    def load_data(self, source: str) -> pl.DataFrame:
        """データを読み込む。

        Args:
            source (str): データソース

        Returns:
            pl.DataFrame: 読み込まれたデータ
        """
        pass

    @abstractmethod
    def save_data(self, data: pl.DataFrame, destination: str) -> None:
        """データを保存する。

        Args:
            data (pl.DataFrame): 保存するデータ
            destination (str): 保存先
        """
        pass

    @abstractmethod
    def get_available_tables(self) -> list[str]:
        """利用可能なテーブル一覧を取得する。

        Returns:
            List[str]: テーブル名のリスト
        """
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> dict[str, Any]:
        """テーブル情報を取得する。

        Args:
            table_name (str): テーブル名

        Returns:
            dict[str, Any]: テーブル情報
        """
        pass
