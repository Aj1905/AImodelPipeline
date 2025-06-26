"""
特徴量変換を行うモジュール

このモジュールは以下の機能を提供します:
- 数値変換(標準化、正規化、対数変換など)
- エンコーディング(ラベルエンコーディング、ワンホットエンコーディングなど)
- 欠損値補充
- データ型の自動検出
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.data.loaders import SQLiteDataLoader


class FeatureTransformer:
    """
    特徴量変換を行うクラス
    """

    def __init__(self):
        """初期化"""
        self.label_encoders = {}
        self.scalers = {}
        self.transformations_applied = []

    def load_data_from_sqlite(self, db_path: Path, table_name: str, columns: list[str]) -> pd.DataFrame:
        """
        SQLiteから指定された列のデータを読み込む

        Args:
            db_path: SQLiteデータベースファイルのパス
            table_name: テーブル名
            columns: 読み込む列名のリスト

        Returns:
            読み込まれたデータのDataFrame
        """
        loader = SQLiteDataLoader()
        df = loader.load_columns(db_path, table_name, columns)
        print(f"\u2713 データを読み込みました (形状: {df.shape})")
        return df

    def detect_data_types(self, df: pd.DataFrame) -> dict[str, str]:
        """
        データフレームの各列のデータ型を自動検出する

        Args:
            df: 対象のDataFrame

        Returns:
            列名とデータ型の辞書
        """
        data_types = {}
        for column in df.columns:
            # 数値型の判定
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].dtype in ["int64", "int32"]:
                    data_types[column] = "integer"
                else:
                    data_types[column] = "float"
            # 日付型の判定
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                data_types[column] = "datetime"
            # カテゴリカル型の判定
            elif df[column].nunique() / len(df) < 0.1:  # ユニーク値が10%未満
                data_types[column] = "categorical"
            # テキスト型の判定
            else:
                data_types[column] = "text"

        return data_types

    def apply_transformations(self, df: pd.DataFrame, transformations_config: dict) -> pd.DataFrame:
        """
        指定された変換を適用する

        Args:
            df: 対象のDataFrame
            transformations_config: 変換設定の辞書

        Returns:
            変換後のDataFrame
        """
        df_processed = df.copy()

        # データ型の自動検出
        data_types = self.detect_data_types(df_processed)
        print(f"📊 検出されたデータ型: {data_types}")

        # 数値変換の適用
        if "numeric_transformations" in transformations_config:
            numeric_columns = [col for col, dtype in data_types.items() if dtype in ["integer", "float"]]
            if numeric_columns:
                df_processed = self.apply_numeric_transformations(
                    df_processed, numeric_columns, transformations_config["numeric_transformations"]
                )

        # カテゴリカル変換の適用
        if "categorical_transformations" in transformations_config:
            categorical_columns = [col for col, dtype in data_types.items() if dtype == "categorical"]
            if categorical_columns:
                df_processed = self.apply_categorical_transformations(
                    df_processed, categorical_columns, transformations_config["categorical_transformations"]
                )

        # 欠損値処理の適用
        if "missing_value_strategy" in transformations_config:
            df_processed = self.handle_missing_values(df_processed, transformations_config["missing_value_strategy"])

        return df_processed

    def apply_numeric_transformations(
        self, df: pd.DataFrame, columns: list[str], transformations: list[str]
    ) -> pd.DataFrame:
        """
        数値列に変換を適用する

        Args:
            df: 対象のDataFrame
            columns: 変換対象の列名リスト
            transformations: 適用する変換のリスト

        Returns:
            変換後のDataFrame
        """
        df_transformed = df.copy()

        for column in columns:
            for transform in transformations:
                if transform == "standardize":
                    scaler = StandardScaler()
                    df_transformed[f"{column}_standardized"] = scaler.fit_transform(df_transformed[[column]])
                    self.scalers[f"{column}_standardized"] = scaler

                elif transform == "normalize":
                    scaler = MinMaxScaler()
                    df_transformed[f"{column}_normalized"] = scaler.fit_transform(df_transformed[[column]])
                    self.scalers[f"{column}_normalized"] = scaler

                elif transform == "log":
                    # 対数変換(負の値を避けるため)
                    if (df_transformed[column] > 0).all():
                        df_transformed[f"{column}_log"] = np.log(df_transformed[column])
                    else:
                        print(f"⚠️  {column}: 負の値が含まれているため対数変換をスキップ")

                elif transform == "sqrt":
                    # 平方根変換(負の値を避けるため)
                    if (df_transformed[column] >= 0).all():
                        df_transformed[f"{column}_sqrt"] = np.sqrt(df_transformed[column])
                    else:
                        print(f"⚠️  {column}: 負の値が含まれているため平方根変換をスキップ")

        return df_transformed

    def apply_categorical_transformations(
        self, df: pd.DataFrame, columns: list[str], transformations: list[str]
    ) -> pd.DataFrame:
        """
        カテゴリカル列に変換を適用する

        Args:
            df: 対象のDataFrame
            columns: 変換対象の列名リスト
            transformations: 適用する変換のリスト

        Returns:
            変換後のDataFrame
        """
        df_transformed = df.copy()

        for column in columns:
            for transform in transformations:
                if transform == "label_encoding":
                    le = LabelEncoder()
                    df_transformed[f"{column}_encoded"] = le.fit_transform(df_transformed[column].astype(str))
                    self.label_encoders[f"{column}_encoded"] = le

                elif transform == "one_hot_encoding":
                    # ワンホットエンコーディング
                    dummies = pd.get_dummies(df_transformed[column], prefix=column)
                    df_transformed = pd.concat([df_transformed, dummies], axis=1)

        return df_transformed

    def handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        欠損値を処理する

        Args:
            df: 対象のDataFrame
            strategy: 処理戦略 ('drop', 'mean', 'median', 'mode', 'forward_fill')

        Returns:
            処理後のDataFrame
        """
        df_processed = df.copy()

        if strategy == "drop":
            df_processed = df_processed.dropna()
        elif strategy == "mean":
            df_processed = df_processed.fillna(df_processed.mean())
        elif strategy == "median":
            df_processed = df_processed.fillna(df_processed.median())
        elif strategy == "mode":
            df_processed = df_processed.fillna(df_processed.mode().iloc[0])
        elif strategy == "forward_fill":
            df_processed = df_processed.fillna(method="ffill")

        return df_processed

    def save_transformed_data(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        変換後のデータを保存する

        Args:
            df: 保存するDataFrame
            output_path: 保存先のパス
        """
        try:
            df.to_csv(output_path, index=False)
            print(f"✓ 変換後のデータを保存しました: {output_path}")
        except Exception as e:
            print(f"❌ データ保存エラー: {e}")

    def get_transformation_summary(self) -> dict:
        """
        適用された変換の要約を取得する

        Returns:
            変換要約の辞書
        """
        return {
            "transformations_applied": self.transformations_applied,
            "label_encoders": list(self.label_encoders.keys()),
            "scalers": list(self.scalers.keys()),
        }
