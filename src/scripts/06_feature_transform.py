#!/usr/bin/env python3
"""
特徴量変換スクリプト

このスクリプトは、SQLiteデータベースのテーブルに対して
特徴量変換を適用するためのツールです。

実行コマンド例:
    python src/scripts/06_feature_transform.py --table TABLE_NAME --columns COL1,COL2 --numeric-transforms standardize,normalize
    python src/scripts/06_feature_transform.py --table TABLE_NAME --categorical-encodings label,onehot --output-table transformed_table
    python src/scripts/06_feature_transform.py --table TABLE_NAME --feature-engineering --keep-date-for-split
    python src/scripts/06_feature_transform.py --help-transforms
    python src/scripts/06_feature_transform.py --table sales_data --columns sales,temperature --numeric-transforms standardize --output-table sales_transformed
    python src/scripts/06_feature_transform.py --table customer_data --categorical-encodings label,onehot --output-table customer_encoded
    python src/scripts/06_feature_transform.py --table weather_data --feature-engineering --keep-date-for-split --output-table weather_features

使用方法:
    python 06_feature_transform.py --table TABLE_NAME --columns COL1,COL2
    python 06_feature_transform.py --help-transforms
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.module.db_utils import (
    select_columns_interactively,
    select_table_interactively,
    validate_columns_exist,
    validate_db_path,
    validate_table_exists,
)
from src.module.feature_transformer import FeatureTransformer, print_transformation_help

# ============================================================================
# 定数定義
# ============================================================================

DEFAULT_DB_PATH = "data/database.sqlite"

# ============================================================================
# 特徴量変換関数
# ============================================================================


def _process_datetime_features(data: pl.DataFrame) -> pl.DataFrame:
    """日時関連の特徴量を処理する"""
    processed_data = data.clone()

    if "date" in data.columns:
        processed_data = processed_data.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
        )

        # Day of week: 0=Monday, 1=Tuesday, ..., 6=Sunday
        processed_data = processed_data.with_columns(
            pl.col("datetime").dt.weekday().alias("dow")
        )

    return processed_data


def _process_time_features(data: pl.DataFrame) -> pl.DataFrame:
    """時間関連の特徴量を処理する"""
    processed_data = data.clone()

    # 給料日が 25 日以降なので、25 日以降を月末として特徴量にする
    if "date_day" in data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("date_day") >= 25).then(1).otherwise(0).alias("is_month_end")
        )

    # 週末
    if "dow" in processed_data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("dow") >= 4).then(1).otherwise(0).alias("is_weekend")
        )

    # ランチタイム・ディナータイム
    if "time" in data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("time").is_in([11, 12, 13]))
            .then(1)
            .otherwise(0)
            .alias("is_lunch")
        )
        processed_data = processed_data.with_columns(
            pl.when(pl.col("time") >= 18).then(1).otherwise(0).alias("is_dinner")
        )

    return processed_data


def _convert_string_columns(
    data: pl.DataFrame, available_columns: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """文字列型の列を数値に変換する"""
    processed_data = data.clone()
    updated_columns = available_columns.copy()

    print("\n🔍 データ型の確認:")
    for col in processed_data.columns:
        dtype = processed_data[col].dtype
        print(f"  {col}: {dtype}")

        # 文字列型の場合は数値に変換を試行
        if dtype == pl.Utf8:
            print(f"    ⚠️  文字列型の列 '{col}' を数値に変換します")
            try:
                # 空文字列をNaNに変換してから数値に変換
                processed_data = processed_data.with_columns(
                    pl.col(col).str.replace("", "null").cast(pl.Float64, strict=False)
                )
                print("    ✅ 変換成功")
            except Exception as e:
                print(f"    ❌ 変換失敗: {e}")
                # 変換できない場合は除外
                if col in updated_columns:
                    updated_columns.remove(col)
                    print(f"    🗑️  列 '{col}' を除外します")

    return processed_data, updated_columns


def _handle_missing_values(data: pl.DataFrame) -> pl.DataFrame:
    """欠損値を処理する"""
    processed_data = data.clone()

    print("\n🔧 欠損値の処理:")
    for col in processed_data.columns:
        null_count = processed_data[col].null_count()
        if null_count > 0:
            print(f"  {col}: {null_count}個の欠損値を0で補完")
            processed_data = processed_data.with_columns(pl.col(col).fill_null(0))

    return processed_data


def feature_engineering(
    data: pl.DataFrame, feature_columns: list[str], keep_date_for_split: bool = False
) -> pl.DataFrame:
    """特徴量エンジニアリングを実行"""
    # 日時関連の特徴量を処理
    processed_data = _process_datetime_features(data)

    # 時間関連の特徴量を処理
    processed_data = _process_time_features(processed_data)

    # 指定された特徴量列のみを選択
    available_columns = [
        col for col in feature_columns if col in processed_data.columns
    ]

    # 時系列分割が必要な場合はdate列を保持(datetime列に変換されても元のdate列を保持)
    if (
        keep_date_for_split
        and "date" in data.columns
        and "date" not in available_columns
    ):
        available_columns.append("date")

    processed_data = processed_data.select(available_columns)

    # 文字列型の列を数値に変換
    processed_data, available_columns = _convert_string_columns(
        processed_data, available_columns
    )

    # 欠損値の処理
    processed_data = _handle_missing_values(processed_data)

    # 最終的な列選択
    final_columns = [col for col in available_columns if col in processed_data.columns]
    processed_data = processed_data.select(final_columns)

    print("\n📊 特徴量の確認:")
    print(processed_data.head())
    print(f"使用する特徴量列: {final_columns}")
    print("最終的なデータ型:")
    for col in processed_data.columns:
        print(f"  {col}: {processed_data[col].dtype}")

    return processed_data


# ============================================================================
# 関数定義
# ============================================================================


def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="SQLiteテーブルの特徴量変換を実行"
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLiteデータベースファイルパス"
    )
    parser.add_argument(
        "--table",
        type=str,
        help="変換対象のテーブル名 (未指定の場合は対話的に選択)"
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="変換対象の列名 (カンマ区切り、未指定の場合は対話的に選択)"
    )
    parser.add_argument(
        "--output-table",
        type=str,
        help="出力テーブル名 (未指定の場合は自動生成)"
    )
    parser.add_argument(
        "--numeric-transforms",
        type=str,
        help="数値変換オプション (カンマ区切り: standardize,normalize,robust_scale,log,sqrt)"
    )
    parser.add_argument(
        "--categorical-encodings",
        type=str,
        help="カテゴリカルエンコーディング (カンマ区切り: label,onehot)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="非対話モードで実行"
    )
    parser.add_argument(
        "--help-transforms",
        action="store_true",
        help="変換オプションのヘルプを表示"
    )
    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        help="特徴量エンジニアリングを実行 (日時特徴量、時間特徴量、文字列変換、欠損値処理)"
    )
    parser.add_argument(
        "--keep-date-for-split",
        action="store_true",
        help="時系列分割用にdate列を保持する"
    )
    return parser.parse_args()


def get_transformation_choices_interactively():
    """対話的に変換オプションを選択する"""
    print("\n" + "=" * 60)
    print("🔧 特徴量変換オプション選択")
    print("=" * 60)

    # 数値変換の選択
    print("\n📊 数値変換オプション:")
    print("1. 標準化 (standardize)")
    print("2. 正規化 (normalize)")
    print("3. ロバストスケーリング (robust_scale)")
    print("4. 対数変換 (log)")
    print("5. 平方根変換 (sqrt)")
    print("6. 数値変換をスキップ")

    choice = input("\n数値変換を選択してください (カンマ区切り、例: 1,3,4): ").strip()

    numeric_transforms = []
    if choice and choice != "6":
        transform_map = {
            "1": "standardize",
            "2": "normalize",
            "3": "robust_scale",
            "4": "log",
            "5": "sqrt"
        }
        numeric_transforms = [
            transform_map.get(x.strip())
            for x in choice.split(",")
            if x.strip() in transform_map
        ]

    # カテゴリカルエンコーディングの選択
    print("\n🏷️  カテゴリカルエンコーディング:")
    print("1. ラベルエンコーディング (label)")
    print("2. ワンホットエンコーディング (onehot)")
    print("3. カテゴリカル変換をスキップ")

    choice = input("\nカテゴリカルエンコーディングを選択してください (カンマ区切り): ").strip()

    categorical_encodings = []
    if choice and choice != "3":
        encoding_map = {"1": "label", "2": "onehot"}
        categorical_encodings = [
            encoding_map.get(x.strip())
            for x in choice.split(",")
            if x.strip() in encoding_map
        ]

    return numeric_transforms, categorical_encodings


def _apply_transformations(
    transformer,
    df,
    numeric_transforms,
    categorical_encodings,
    args
):
    """変換を適用する"""
    transformations_config = {}

    # 数値変換の設定
    if numeric_transforms:
        transformations_config["numeric_transformations"] = numeric_transforms

    # カテゴリカル変換の設定
    if categorical_encodings:
        transformations_config["categorical_transformations"] = categorical_encodings

    # 変換の適用
    if transformations_config:
        # データ型の検出
        data_types = transformer.detect_data_types(df)

        print("\n🔢 数値変換中...")
        numeric_columns = [
            col for col, dtype in data_types.items()
            if dtype == "numeric"
        ]
        if numeric_columns:
            print(f"対象列: {', '.join(numeric_columns)}")

        print("\n🏷️  カテゴリカルエンコーディング中...")
        categorical_columns = [
            col for col, dtype in data_types.items()
            if dtype == "categorical"
        ]
        if categorical_columns:
            print(f"対象列: {', '.join(categorical_columns)}")

        df = transformer.apply_transformations(df, transformations_config)
        print("✅ 変換完了")

    return df


def _setup_database_and_table(args):
    """データベースとテーブルの設定を行う"""
    # データベースパスの検証
    db_path = Path(args.db_file).expanduser().resolve()
    if not validate_db_path(db_path):
        return None, None

    # テーブル名の決定
    table_name = args.table
    if not table_name:
        table_name = select_table_interactively(db_path)
        if not table_name:
            return None, None

    # テーブル存在確認
    if not validate_table_exists(db_path, table_name):
        return None, None

    return db_path, table_name


def _setup_columns(args, db_path, table_name):
    """列名の設定を行う"""
    column_names = []
    if args.columns:
        column_names = [col.strip() for col in args.columns.split(",")]
    else:
        column_names = select_columns_interactively(db_path, table_name)
        if not column_names:
            return None

    # 列存在確認
    if not validate_columns_exist(db_path, table_name, column_names):
        return None

    return column_names


def _setup_transformations(args):
    """変換オプションの設定を行う"""
    numeric_transforms = []
    categorical_encodings = []

    if args.non_interactive:
        if args.numeric_transforms:
            numeric_transforms = [t.strip() for t in args.numeric_transforms.split(",")]
        if args.categorical_encodings:
            categorical_encodings = [e.strip() for e in args.categorical_encodings.split(",")]
    else:
        numeric_transforms, categorical_encodings = get_transformation_choices_interactively()

    return numeric_transforms, categorical_encodings


def _process_data(db_path, table_name, column_names, numeric_transforms, categorical_encodings, args):
    """データの読み込みと変換を行う"""
    # データ読み込み
    print(f"\n📊 データ読み込み中: {table_name}")

    # Polarsを使用してデータを読み込み
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pl.read_database(f"SELECT * FROM {table_name}", conn)
    conn.close()

    print(f"読み込み完了: {df.shape[0]}行 x {df.shape[1]}列")

    # 特徴量エンジニアリングの実行
    if args.feature_engineering:
        print("\n🔧 特徴量エンジニアリング実行中...")
        df = feature_engineering(df, column_names, args.keep_date_for_split)
        print("✅ 特徴量エンジニアリング完了")

    # 従来の変換の適用
    transformer = FeatureTransformer()
    df = _apply_transformations(
        transformer,
        df,
        numeric_transforms,
        categorical_encodings,
        args
    )

    return df, transformer


def _save_and_display_results(db_path, table_name, output_table, df, transformer, column_names):
    """結果の保存と表示を行う"""
    # データベースに保存
    print(f"\n💾 変換結果を保存中: {output_table}")

    # Polarsを使用してSQLiteに保存
    import sqlite3
    conn = sqlite3.connect(db_path)
    df.write_database(output_table, conn, if_exists="replace")
    conn.close()

    print("✅ 保存完了")

    # 結果表示
    print("\n📋 変換結果:")
    print(f"  元テーブル: {table_name}")
    print(f"  出力テーブル: {output_table}")
    print(f"  変換前: {len(column_names)}列")
    print(f"  変換後: {df.shape[1]}列")
    print(f"  行数: {df.shape[0]}行")

    if transformer.transformations_applied:
        print(f"  適用された変換: {', '.join(transformer.transformations_applied)}")


def main():
    """メイン関数"""
    args = parse_arguments()

    # ヘルプ表示
    if args.help_transforms:
        print_transformation_help()
        return

    # データベースとテーブルの設定
    db_path, table_name = _setup_database_and_table(args)
    if db_path is None:
        return

    # 列名の設定
    column_names = _setup_columns(args, db_path, table_name)
    if column_names is None:
        return

    # 変換オプションの設定
    numeric_transforms, categorical_encodings = _setup_transformations(args)

    # データの処理
    df, transformer = _process_data(
        db_path, table_name, column_names,
        numeric_transforms, categorical_encodings, args
    )

    # 出力テーブル名の決定
    output_table = args.output_table
    if not output_table:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_table = f"{table_name}_transformed_{timestamp}"

    # 結果の保存と表示
    _save_and_display_results(db_path, table_name, output_table, df, transformer, column_names)


if __name__ == "__main__":
    main()
