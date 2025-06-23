#!/usr/bin/env python3
"""
SQLiteファイルを読み込み、特徴量変換を行うスクリプト

このスクリプトは以下の機能を提供します:
- 数値変換(標準化、正規化、対数変換など)
- エンコーディング(ラベルエンコーディング、ワンホットエンコーディングなど)
- 欠損値補充
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ml.additional_module.db_utils import (
    select_columns_interactively,
    select_table_interactively,
    validate_columns_exist,
    validate_db_path,
    validate_table_exists,
)
from src.ml.feature_transformer import FeatureTransformer, print_transformation_help

# ============================================================================
# メイン処理関数
# ============================================================================


def _parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="SQLiteファイルの特徴量変換")
    parser.add_argument(
        "--db-file",
        type=str,
        default="data/database.sqlite",
        help="SQLiteデータベースファイルパス (デフォルト: data/database.sqlite)",
    )
    parser.add_argument("--table", type=str, help="変換対象のテーブル名 (未指定の場合は対話的に選択)")
    parser.add_argument("--columns", type=str, help="変換対象の列名 (カンマ区切り、未指定の場合は対話的に選択)")
    parser.add_argument("--output-table", type=str, help="出力テーブル名 (未指定の場合は自動生成)")
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["auto", "mean", "median", "mode", "drop"],
        default="auto",
        help="欠損値処理戦略 (デフォルト: auto)",
    )
    parser.add_argument(
        "--numeric-transforms",
        type=str,
        nargs="*",
        choices=["standardize", "normalize", "robust_scale", "log", "sqrt"],
        help="数値変換の種類",
    )
    parser.add_argument(
        "--categorical-encodings",
        type=str,
        nargs="*",
        choices=["label", "onehot"],
        help="カテゴリカルエンコーディングの種類",
    )
    parser.add_argument("--non-interactive", action="store_true", help="非対話モードで実行")
    parser.add_argument("--help-transforms", action="store_true", help="変換オプションのヘルプを表示")

    return parser.parse_args()


def _select_table_and_columns(args, db_path):
    """テーブルと列を選択する"""
    # テーブル選択
    if args.table:
        table_name = args.table
        if not validate_table_exists(db_path, table_name):
            print(f"❌ エラー: テーブル '{table_name}' が存在しません")
            return None, None
    else:
        table_name = select_table_interactively(db_path)
        if not table_name:
            print("テーブルが選択されませんでした")
            return None, None

    print(f"📋 選択されたテーブル: {table_name}")

    # 列選択
    if args.columns:
        selected_columns = [col.strip() for col in args.columns.split(",")]
        # 列の存在確認
        columns_exist, missing_columns = validate_columns_exist(db_path, table_name, selected_columns)
        if not columns_exist:
            print(f"❌ エラー: 以下の列が見つかりません: {missing_columns}")
            return None, None
    else:
        selected_columns = select_columns_interactively(db_path, table_name)
        if not selected_columns:
            print("列が選択されませんでした")
            return None, None

    print(f"📊 選択された列: {', '.join(selected_columns)}")
    return table_name, selected_columns


def _select_transformations(args):
    """変換オプションを選択する"""
    if args.non_interactive:
        numeric_transforms = args.numeric_transforms or []
        categorical_encodings = args.categorical_encodings or []
    else:
        numeric_transforms = []
        categorical_encodings = []

        # 数値変換の選択
        print("\n数値変換オプション:")
        print("  1. 標準化 (standardize)")
        print("  2. 正規化 (normalize)")
        print("  3. ロバストスケーリング (robust_scale)")
        print("  4. 対数変換 (log)")
        print("  5. 平方根変換 (sqrt)")
        print("  6. スキップ")

        choice = input("数値変換を選択してください (カンマ区切り、例: 1,2,4): ").strip()
        if choice and choice != "6":
            transform_map = {"1": "standardize", "2": "normalize", "3": "robust_scale", "4": "log", "5": "sqrt"}
            numeric_transforms = [transform_map.get(x.strip()) for x in choice.split(",") if x.strip() in transform_map]

        # カテゴリカルエンコーディングの選択
        print("\nカテゴリカルエンコーディングオプション:")
        print("  1. ラベルエンコーディング (label)")
        print("  2. ワンホットエンコーディング (onehot)")
        print("  3. スキップ")

        choice = input("カテゴリカルエンコーディングを選択してください (カンマ区切り、例: 1,2): ").strip()
        if choice and choice != "3":
            encoding_map = {"1": "label", "2": "onehot"}
            categorical_encodings = [
                encoding_map.get(x.strip()) for x in choice.split(",") if x.strip() in encoding_map
            ]

    return numeric_transforms, categorical_encodings


def _apply_transformations(transformer, df, numeric_transforms, categorical_encodings, args):
    """変換を適用する"""
    # 欠損値処理
    print(f"\n🔧 欠損値処理中... (戦略: {args.missing_strategy})")
    df = transformer.handle_missing_values(df, args.missing_strategy)
    print("✓ 欠損値処理完了")

    # データ型の検出
    data_types = transformer.detect_data_types(df)
    print("\n📊 データ型検出結果:")
    for col, dtype in data_types.items():
        print(f"  {col}: {dtype}")

    # 数値変換
    if numeric_transforms:
        print("\n🔢 数値変換中...")
        numeric_columns = [col for col, dtype in data_types.items() if dtype == "numeric"]
        if numeric_columns:
            df = transformer.apply_numeric_transformations(df, numeric_columns, numeric_transforms)
        else:
            print("数値列が見つかりませんでした")

    # カテゴリカルエンコーディング
    if categorical_encodings:
        print("\n🏷️  カテゴリカルエンコーディング中...")
        categorical_columns = [col for col, dtype in data_types.items() if dtype == "categorical"]
        if categorical_columns:
            df = transformer.apply_categorical_encodings(df, categorical_columns, categorical_encodings)
        else:
            print("カテゴリカル列が見つかりませんでした")

    return df


def main():
    """メイン関数"""
    args = _parse_arguments()

    # ヘルプ表示
    if args.help_transforms:
        print_transformation_help()
        return

    # パス解決
    db_path = Path(args.db_file).expanduser().resolve()

    # データベース存在チェック
    if not validate_db_path(db_path):
        return

    print(f"🗃️  データベース: {db_path}")

    # テーブルと列の選択
    table_name, selected_columns = _select_table_and_columns(args, db_path)
    if not table_name or not selected_columns:
        return

    # 変換オプションの選択
    numeric_transforms, categorical_encodings = _select_transformations(args)

    # 特徴量変換の実行
    transformer = FeatureTransformer()

    print("\n🔄 データ読み込み中...")
    df = transformer.load_data_from_sqlite(db_path, table_name, selected_columns)
    print(f"✓ データ読み込み完了: {len(df)}行, {len(df.columns)}列")

    # 変換の適用
    df = _apply_transformations(transformer, df, numeric_transforms, categorical_encodings, args)

    # 結果の保存
    output_table = args.output_table or f"{table_name}_transformed"
    print("\n💾 変換結果を保存中...")
    transformer.save_transformed_data(df, db_path, output_table)

    # 変換要約の表示
    summary = transformer.get_transformation_summary()
    print("\n📈 変換要約:")
    print(f"  スケーラー: {len(summary['scalers'])}個")
    print(f"  エンコーダー: {len(summary['encoders'])}個")
    print(f"  補完器: {len(summary['imputers'])}個")
    print(f"  総変換数: {summary['total_transformations']}個")

    print("\n✅ 特徴量変換が完了しました!")
    print(f"元テーブル: {table_name}")
    print(f"出力テーブル: {output_table}")
    print(f"変換前: {len(selected_columns)}列")
    print(f"変換後: {len(df.columns)}列")


if __name__ == "__main__":
    main()
