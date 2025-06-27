from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureManager:
    """Manage feature transformations"""

    features: pd.DataFrame | None = None
    categorical_columns: list[str] | None = None
    date_columns: list[str] | None = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features to be compatible with LightGBM."""
        self.features = df.copy()

        # データ型の情報を表示
        print("🔍 データ型の確認:")
        print(self.features.dtypes.value_counts())

        # 日付列を特定
        self.date_columns = self._identify_date_columns()
        if self.date_columns:
            print(f"📅 日付列を検出: {self.date_columns}")
            self._process_date_columns()

        # カテゴリカル列を特定
        self.categorical_columns = self._identify_categorical_columns()
        if self.categorical_columns:
            print(f"🏷️ カテゴリカル列を検出: {self.categorical_columns}")
            self._process_categorical_columns()

        # 数値列の処理
        self._process_numeric_columns()

        # 最終的なデータ型を確認
        print("✅ 前処理後のデータ型:")
        print(self.features.dtypes.value_counts())

        return self.features

    def _identify_date_columns(self) -> list[str]:
        """日付列を特定する"""
        date_columns = []
        for col in self.features.columns:
            if self.features[col].dtype == 'object':
                # 最初の非NaN値をチェック
                sample = self.features[col].dropna().iloc[0] if len(self.features[col].dropna()) > 0 else None
                if sample and isinstance(sample, str):
                    # 日付形式の文字列かどうかをチェック
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime']):
                        date_columns.append(col)
        return date_columns

    def _identify_categorical_columns(self) -> list[str]:
        """カテゴリカル列を特定する"""
        categorical_columns = []
        for col in self.features.columns:
            if self.features[col].dtype == 'object':
                # 日付列でない場合
                if col not in (self.date_columns or []):
                    # ユニーク値の数が少ない場合(カテゴリカルとみなす)
                    unique_count = self.features[col].nunique()
                    if unique_count <= 50:  # 閾値は調整可能
                        categorical_columns.append(col)
        return categorical_columns

    def _process_date_columns(self) -> None:
        """日付列を数値特徴量に変換する"""
        for col in self.date_columns:
            try:
                # 日付文字列をdatetimeに変換
                self.features[col] = pd.to_datetime(self.features[col], errors='coerce')

                # 日付特徴量を抽出
                self.features[f'{col}_year'] = self.features[col].dt.year
                self.features[f'{col}_month'] = self.features[col].dt.month
                self.features[f'{col}_day'] = self.features[col].dt.day
                self.features[f'{col}_dayofweek'] = self.features[col].dt.dayofweek

                # 元の列を削除
                self.features = self.features.drop(columns=[col])

            except Exception as e:
                print(f"⚠️ 日付列 '{col}' の処理でエラー: {e}")
                # エラーが発生した場合は列を削除
                self.features = self.features.drop(columns=[col])

    def _process_categorical_columns(self) -> None:
        """カテゴリカル列を数値にエンコードする"""
        for col in self.categorical_columns:
            try:
                # カテゴリカルエンコーディング
                self.features[col] = pd.Categorical(self.features[col]).codes
                # -1(欠損値)をNaNに変換
                self.features[col] = self.features[col].replace(-1, np.nan)
            except Exception as e:
                print(f"⚠️ カテゴリカル列 '{col}' の処理でエラー: {e}")
                # エラーが発生した場合は列を削除
                self.features = self.features.drop(columns=[col])

    def _process_numeric_columns(self) -> None:
        """数値列の処理"""
        # 数値列のみを残す
        numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        non_numeric_columns = self.features.select_dtypes(exclude=[np.number]).columns

        if len(non_numeric_columns) > 0:
            print(f"⚠️ 数値以外の列を削除: {list(non_numeric_columns)}")
            self.features = self.features[numeric_columns]

        # 無限大の値をNaNに変換(LightGBMはNaNを適切に処理できる)
        self.features = self.features.replace([np.inf, -np.inf], np.nan)

        # 欠損値はLightGBMが適切に処理します(欠損値の数: {self.features.isnull().sum().sum()})
