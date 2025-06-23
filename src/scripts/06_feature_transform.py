#!/usr/bin/env python3
"""
特徴量変換スクリプト

このスクリプトは、SQLiteデータベースのテーブルに対して
特徴量変換を適用するためのツールです。

使用方法:
    python 06_feature_transform.py --table TABLE_NAME --columns COL1,COL2
    python 06_feature_transform.py --help-transforms
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

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
    transformer = FeatureTransformer()
    df = transformer.load_data_from_sqlite(db_path, table_name, column_names)
    print(f"読み込み完了: {df.shape[0]}行 x {df.shape[1]}列")

    # 変換の適用
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
    transformer.save_to_sqlite(db_path, output_table, df)
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
