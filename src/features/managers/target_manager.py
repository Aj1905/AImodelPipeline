"""
ターゲット変数の管理クラス。
"""

import pandas as pd
import polars as pl


class TargetManager:
    """
    ターゲット変数を管理するクラス。
    """

    def __init__(self, target_data: pl.Series):
        """
        初期化。

        Args:
            target_data: ターゲットデータ
        """
        # 空の名前をチェック
        if not target_data.name or target_data.name.strip() == "":
            raise ValueError("ターゲット名は空文字列にできません")

        # 空のデータをチェック
        if len(target_data) == 0:
            raise ValueError("ターゲットデータは空のリストにできません")

        self._target_data = target_data

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, target_column: str) -> "TargetManager":
        """
        DataFrameから指定された列を使ってTargetManagerを作成。

        Args:
            df: 入力DataFrame
            target_column: ターゲット列名

        Returns:
            TargetManager: 新しいTargetManagerインスタンス
        """
        if target_column not in df.columns:
            raise ValueError(f"列 '{target_column}' がDataFrameに存在しません")

        target_series = df[target_column]
        return cls(target_data=target_series)

    def get_target_data(self) -> list:
        """ターゲットデータをリストとして取得"""
        if self._target_data is None:
            return []
        return self._target_data.to_list()

    def get_target_name(self) -> str:
        """ターゲット名を取得。"""
        if self._target_data is None:
            return "target"
        return self._target_data.name or "target"

    def to_polars_series(self) -> pl.Series:
        """Polars Seriesとして取得。"""
        return self._target_data

    def update_from_dataframe(self, df: pl.DataFrame, target_column: str) -> None:
        """
        DataFrameからターゲットデータを更新。

        Args:
            df: 入力DataFrame
            target_column: ターゲット列名
        """
        if target_column not in df.columns:
            raise ValueError(f"列 '{target_column}' がDataFrameに存在しません")

        self._target_data = df[target_column]

    def update_from_series(self, series: pl.Series) -> None:
        """
        Seriesからターゲットデータを更新。

        Args:
            series: 入力Series
        """
        if not isinstance(series, pl.Series):
            raise ValueError("seriesはpl.Seriesである必要があります")

        self._target_data = series

    def is_empty(self) -> bool:
        """ターゲットデータが空かどうかを判定。"""
        return len(self._target_data) == 0

    def __len__(self) -> int:
        """ターゲットデータの長さを返す。"""
        return len(self._target_data)

    def __repr__(self) -> str:
        """文字列表現を返す。"""
        target_name = self.get_target_name()
        return f"TargetManager(target_name='{target_name}', length={len(self)})"

    def __str__(self) -> str:
        """ターゲットマネージャーの詳細な文字列表現を返す。"""
        target_name = self.get_target_name()
        data_length = len(self)

        if data_length == 0:
            return f"🎯 TargetManager: {target_name} (データなし)"

        # 基本統計を計算
        try:
            target_data = self._target_data
            mean_val = target_data.mean()
            std_val = target_data.std()
            min_val = target_data.min()
            max_val = target_data.max()

            result = "🎯 TargetManager:\n"
            result += f"  ターゲット名: {target_name}\n"
            result += f"  データ長: {data_length}\n"
            result += f"  平均: {mean_val:.4f}\n"
            result += f"  標準偏差: {std_val:.4f}\n"
            result += f"  最小値: {min_val:.4f}\n"
            result += f"  最大値: {max_val:.4f}"

            return result
        except Exception:
            return f"🎯 TargetManager: {target_name} (長さ: {data_length})"

    def get_target(self, target_name: str) -> pd.Series:
        """ターゲットを取得する(カラム名)"""
        # Implementation of get_target method
        pass 