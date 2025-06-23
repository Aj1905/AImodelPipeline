"""
特徴量管理クラス。

特徴量エンジニアリングの結果を格納し、モデルが受け付ける形式に変換する機能を提供します。
"""

from typing import Any

import pandas as pd
import polars as pl


class FeatureManager:
    """特徴量を管理するクラス。"""

    def __init__(self, initial_features: pl.DataFrame | None = None):
        """FeatureManagerを初期化する。

        Args:
            initial_features (pl.DataFrame | None, optional): 初期特徴量DataFrame. デフォルトは None.
        """
        self._features: pl.DataFrame | None = None
        self._feature_flags: dict[str, bool] = {}

        if initial_features is not None:
            self._features = initial_features.clone()
            for col in initial_features.columns:
                self._feature_flags[col] = True

    def add_features(self, df: pl.DataFrame, enabled: bool = True) -> None:
        """DataFrameから特徴量を追加する。

        Args:
            df (pl.DataFrame): 特徴量を含むDataFrame
            enabled (bool, optional): 特徴量を有効にするかどうか. デフォルトは True.

        Raises:
            ValueError: DataFrameが空の場合
            ValueError: 特徴量の長さが他の特徴量と一致しない場合
        """
        # 空の列名をチェック
        for col in df.columns:
            if not col or col.strip() == "":
                raise ValueError("特徴量名は空文字列にできません")

        self._validate_dataframe_length(df)

        # DataFrameに列を追加
        if self._features is None:
            self._features = df.clone()
        else:
            # 既存の特徴量と結合
            for col in df.columns:
                self._features = self._features.with_columns(df[col])

        # フラグを設定
        for col in df.columns:
            self._feature_flags[col] = enabled

    def remove_feature(self, name: str) -> None:
        """特徴量を削除する。

        Args:
            name (str): 削除する特徴量の名称

        Raises:
            KeyError: 指定された特徴量が存在しない場合
        """
        if self._features is None or name not in self._features.columns:
            raise KeyError(f"特徴量 '{name}' が見つかりません")

        # DataFrameから列を削除
        self._features = self._features.drop(name)
        del self._feature_flags[name]

    def enable_feature(self, name: str) -> None:
        """特徴量を有効にする。

        Args:
            name (str): 有効にする特徴量の名称

        Raises:
            KeyError: 指定された特徴量が存在しない場合
        """
        if self._features is None or name not in self._features.columns:
            raise KeyError(f"特徴量 '{name}' が見つかりません")

        self._feature_flags[name] = True

    def disable_feature(self, name: str) -> None:
        """特徴量を無効にする。

        Args:
            name (str): 無効にする特徴量の名称

        Raises:
            KeyError: 指定された特徴量が存在しない場合
        """
        if self._features is None or name not in self._features.columns:
            raise KeyError(f"特徴量 '{name}' が見つかりません")

        self._feature_flags[name] = False

    def get_enabled_features(self) -> pl.DataFrame:
        """有効な特徴量のみをDataFrameとして取得する。

        Returns:
            pl.DataFrame: 有効な特徴量を含むDataFrame
        """
        if self._features is None:
            return pl.DataFrame()

        enabled_columns = [col for col in self._features.columns if self._feature_flags.get(col, False)]
        if not enabled_columns:
            return pl.DataFrame()

        return self._features.select(enabled_columns)

    def get_all_features(self) -> pl.DataFrame:
        """すべての特徴量をDataFrameとして取得する。

        Returns:
            pl.DataFrame: すべての特徴量を含むDataFrame
        """
        if self._features is None:
            return pl.DataFrame()

        return self._features.clone()

    def get_feature_flags(self) -> dict[str, bool]:
        """特徴量の有効/無効フラグを取得する。

        Returns:
            dict[str, bool]: 特徴量フラグの辞書
        """
        return self._feature_flags.copy()

    def set_feature_flags(self, flags: dict[str, bool]) -> None:
        """特徴量の有効/無効フラグを一括設定する。

        Args:
            flags (dict[str, bool]): 設定するフラグの辞書

        Raises:
            KeyError: 存在しない特徴量のフラグを設定しようとした場合
        """
        for name, flag in flags.items():
            if self._features is None or name not in self._features.columns:
                raise KeyError(f"特徴量 '{name}' が見つかりません")
            self._feature_flags[name] = flag

    def to_polars_dataframe(self, use_enabled_only: bool = True) -> pl.DataFrame:
        """特徴量をPolars DataFrameとして取得する。

        Args:
            use_enabled_only (bool, optional): 有効な特徴量のみを使用するかどうか. デフォルトは True.

        Returns:
            pl.DataFrame: 特徴量を含むDataFrame

        Raises:
            ValueError: 特徴量が空の場合
        """
        if use_enabled_only:
            result = self.get_enabled_features()
        else:
            result = self.get_all_features()

        if result.is_empty():
            raise ValueError("変換する特徴量がありません")

        return result

    def update_features_from_dataframe(self, df: pl.DataFrame, feature_names: list[str] | None = None) -> None:
        """DataFrameから特徴量を更新する。

        Args:
            df (pl.DataFrame): 特徴量を含むDataFrame
            feature_names (list[str] | None, optional): 更新する特徴量名のリスト. Noneの場合はすべての列を使用.
        """
        columns = feature_names if feature_names is not None else df.columns

        # 既存の特徴量を削除
        for col in columns:
            if self._features is not None and col in self._features.columns:
                self.remove_feature(col)

        # 新しい特徴量を追加
        selected_df = df.select([col for col in columns if col in df.columns])
        if not selected_df.is_empty():
            self.add_features(selected_df, enabled=True)

    def get_feature_count(self) -> int:
        """特徴量の総数を取得する。

        Returns:
            int: 特徴量の総数
        """
        if self._features is None:
            return 0
        return len(self._features.columns)

    def get_enabled_feature_count(self) -> int:
        """有効な特徴量の数を取得する。

        Returns:
            int: 有効な特徴量の数
        """
        return sum(1 for flag in self._feature_flags.values() if flag)

    def __len__(self) -> int:
        """特徴量データの長さを取得する。

        Returns:
            int: データの長さ(0の場合は特徴量が空)
        """
        if self._features is None:
            return 0
        return self._features.height

    def __repr__(self) -> str:
        """オブジェクトの文字列表現を返す。

        Returns:
            str: オブジェクトの文字列表現
        """
        enabled_count = self.get_enabled_feature_count()
        total_count = self.get_feature_count()
        return f"FeatureManager(features={total_count}, enabled={enabled_count})"

    def __str__(self) -> str:
        """特徴量マネージャーの詳細な文字列表現を返す。

        Returns:
            str: 詳細な文字列表現
        """
        enabled_count = self.get_enabled_feature_count()
        total_count = self.get_feature_count()
        data_length = len(self)

        if total_count == 0:
            return "🔧 FeatureManager: 特徴量なし"

        enabled_features = [col for col in self._features.columns if self._feature_flags.get(col, False)]
        disabled_features = [col for col in self._features.columns if not self._feature_flags.get(col, False)]

        result = "🔧 FeatureManager:\n"
        result += f"  総特徴量数: {total_count}\n"
        result += f"  有効特徴量数: {enabled_count}\n"
        result += f"  データ長: {data_length}\n"

        if enabled_features:
            result += f"  有効特徴量: {', '.join(enabled_features[:10])}"
            if len(enabled_features) > 10:
                result += f" ... (+{len(enabled_features) - 10}個)"
            result += "\n"

        if disabled_features:
            result += f"  無効特徴量: {', '.join(disabled_features[:5])}"
            if len(disabled_features) > 5:
                result += f" ... (+{len(disabled_features) - 5}個)"

        return result.rstrip()

    def _validate_dataframe_length(self, df: pl.DataFrame) -> None:
        """DataFrameの長さを検証する。

        Args:
            df (pl.DataFrame): 検証するDataFrame

        Raises:
            ValueError: DataFrameの長さが既存のデータと一致しない場合
        """
        if df.is_empty():
            raise ValueError("特徴量値は空のリストにできません")

        if self._features is None:
            return

        if df.height != self._features.height:
            raise ValueError(
                f"DataFrameの長さ ({df.height}) が既存のデータの長さ ({self._features.height}) と一致しません"
            )

    def get_feature_importance(self, model: Any) -> pd.DataFrame:
        """特徴量の重要度を取得する(モデル依存)"""
        # Implementation of get_feature_importance method
        pass
