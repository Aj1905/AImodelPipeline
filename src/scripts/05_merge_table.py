#!/usr/bin/env python3
"""
テーブル結合スクリプト

このスクリプトは、SQLiteデータベースの複数のテーブルを結合するためのツールです。

使用方法:
    python 05_merge_table.py --tables TABLE1,TABLE2,TABLE3
    python 05_merge_table.py --auto-merge
    python 05_merge_table.py --help
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.handlers.sqlite_handler import SQLiteHandler
from src.data.utils.interactive_selector import (
    get_available_tables,
    validate_table_exists,
)

# ============================================================================
# 定数定義
# ============================================================================

DEFAULT_DB_PATH = "data/database.sqlite"

# ============================================================================
# テーブル情報取得関数
# ============================================================================


def get_table_info(db_path: Path, table_name: str) -> dict[str, Any]:
    """テーブルの詳細情報を取得する"""
    handler = SQLiteHandler(db_path)

    # テーブルの行数と列数を取得
    row_count = handler.fetch_one(f"SELECT COUNT(*) FROM {table_name}")[0]
    columns_info = handler.get_table_info(table_name)
    column_names = [col[1] for col in columns_info]
    column_types = [col[2] for col in columns_info]

    return {
        "name": table_name,
        "row_count": row_count,
        "column_count": len(column_names),
        "columns": column_names,
        "column_types": column_types
    }


def get_tables_info(db_path: Path, table_names: list[str]) -> list[dict[str, Any]]:
    """複数のテーブルの情報を取得する"""
    tables_info = []
    for table_name in table_names:
        if validate_table_exists(db_path, table_name):
            info = get_table_info(db_path, table_name)
            tables_info.append(info)
        else:
            print(f"⚠️  テーブル '{table_name}' が存在しません")

    return tables_info


def display_tables_info(tables_info: list[dict[str, Any]]) -> None:
    """テーブル情報を表示する"""
    print("\n📊 テーブル情報:")
    print("=" * 80)

    for i, info in enumerate(tables_info, 1):
        print(f"\n{i}. {info['name']}")
        print(f"   行数: {info['row_count']:,}")
        print(f"   列数: {info['column_count']}")
        print(f"   列名: {', '.join(info['columns'])}")


# ============================================================================
# 結合戦略決定関数
# ============================================================================


def determine_merge_strategy(tables_info: list[dict[str, Any]]) -> str:
    """結合戦略を決定する"""
    if len(tables_info) < 2:
        return "none"

    # 全てのテーブルの列名を取得
    all_columns = [set(info['columns']) for info in tables_info]

    # 列名が全て同じかチェック
    common_columns = set.intersection(*all_columns)
    if len(common_columns) == len(tables_info[0]['columns']):
        print("✅ 全てのテーブルで列名が同じです → 縦結合(UNION)を推奨")
        return "vertical"

    # 行数が全て同じかチェック
    row_counts = [info['row_count'] for info in tables_info]
    if len(set(row_counts)) == 1:
        print("✅ 全てのテーブルで行数が同じです → 横結合(JOIN)を推奨")
        return "horizontal"

    # 列名が部分的に同じ場合
    if len(common_columns) > 0:
        print(f"⚠️  共通の列名があります: {', '.join(common_columns)}")
        print("   手動で結合方法を選択してください")
        return "manual"

    print("⚠️  列名も行数も異なります。手動で結合方法を選択してください")
    return "manual"


def get_merge_strategy_interactively(tables_info: list[dict[str, Any]]) -> str:
    """対話的に結合戦略を選択する"""
    print("\n🔧 結合方法の選択")
    print("=" * 40)

    # 自動判定
    auto_strategy = determine_merge_strategy(tables_info)

    if auto_strategy == "none":
        return "none"

    print(f"\n推奨される結合方法: {auto_strategy}")

    print("\n選択可能な結合方法:")
    print("1. 縦結合(UNION) - 列名が同じテーブルを縦に結合")
    print("2. 横結合(JOIN) - 行数が同じテーブルを横に結合")
    print("3. 手動結合 - 結合キーを指定して結合")
    print("4. 結合をキャンセル")

    while True:
        choice = input("\n結合方法を選択してください (1-4): ").strip()

        if choice == "1":
            return "vertical"
        elif choice == "2":
            return "horizontal"
        elif choice == "3":
            return "manual"
        elif choice == "4":
            return "cancel"
        else:
            print("1から4の間で選択してください")


# ============================================================================
# データ読み込み関数
# ============================================================================


def load_table_data(db_path: Path, table_name: str) -> pl.DataFrame:
    """テーブルデータを読み込む"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pl.read_database(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def load_tables_data(db_path: Path, table_names: list[str]) -> list[pl.DataFrame]:
    """複数のテーブルデータを読み込む"""
    dataframes = []
    for table_name in table_names:
        print(f"📥 {table_name} を読み込み中...")
        df = load_table_data(db_path, table_name)
        dataframes.append(df)
        print(f"   ✅ {df.shape[0]}行 x {df.shape[1]}列")

    return dataframes


# ============================================================================
# 結合実行関数
# ============================================================================


def merge_vertically(dataframes: list[pl.DataFrame], table_names: list[str]) -> pl.DataFrame:
    """縦結合(UNION)を実行する"""
    print("\n🔄 縦結合(UNION)実行中...")

    # 最初のデータフレームをベースにする
    merged_df = dataframes[0].clone()

    # 残りのデータフレームを縦に結合
    for i, df in enumerate(dataframes[1:], 1):
        print(f"  {table_names[0]} + {table_names[i]} を結合中...")
        merged_df = pl.concat([merged_df, df], how="vertical")

    print(f"✅ 縦結合完了: {merged_df.shape[0]}行 x {merged_df.shape[1]}列")
    return merged_df


def merge_horizontally(dataframes: list[pl.DataFrame], table_names: list[str]) -> pl.DataFrame:
    """横結合(JOIN)を実行する"""
    print("\n🔄 横結合(JOIN)実行中...")

    # 最初のデータフレームをベースにする
    merged_df = dataframes[0].clone()

    # 残りのデータフレームを横に結合
    for i, df in enumerate(dataframes[1:], 1):
        print(f"  {table_names[0]} + {table_names[i]} を結合中...")
        # インデックスベースで結合
        merged_df = pl.concat([merged_df, df], how="horizontal")

    print(f"✅ 横結合完了: {merged_df.shape[0]}行 x {merged_df.shape[1]}列")
    return merged_df


def merge_manually(dataframes: list[pl.DataFrame], table_names: list[str], tables_info: list[dict[str, Any]]) -> pl.DataFrame:
    """手動結合を実行する"""
    print("\n🔄 手動結合実行中...")

    # 共通の列名を表示
    all_columns = [set(info['columns']) for info in tables_info]
    common_columns = set.intersection(*all_columns)

    if common_columns:
        print(f"共通の列名: {', '.join(common_columns)}")
        join_key = input("結合キーとして使用する列名を入力してください: ").strip()

        if join_key in common_columns:
            # 結合キーを使用して結合
            merged_df = dataframes[0].clone()
            for i, df in enumerate(dataframes[1:], 1):
                print(f"  {table_names[0]} + {table_names[i]} を結合中...")
                merged_df = merged_df.join(df, on=join_key, how="outer")

            print(f"✅ 手動結合完了: {merged_df.shape[0]}行 x {merged_df.shape[1]}列")
            return merged_df

    print("❌ 有効な結合キーが見つかりませんでした")
    return None


# ============================================================================
# 結果保存関数
# ============================================================================


def save_merged_table(db_path: Path, merged_df: pl.DataFrame, output_table: str) -> None:
    """結合結果をデータベースに保存する"""
    print(f"\n💾 結合結果を保存中: {output_table}")

    import sqlite3
    conn = sqlite3.connect(db_path)
    merged_df.write_database(output_table, conn, if_exists="replace")
    conn.close()

    print("✅ 保存完了")


def display_merge_results(merged_df: pl.DataFrame, table_names: list[str], output_table: str) -> None:
    """結合結果を表示する"""
    print("\n📋 結合結果:")
    print("=" * 60)
    print(f"  入力テーブル: {', '.join(table_names)}")
    print(f"  出力テーブル: {output_table}")
    print(f"  結合後: {merged_df.shape[0]}行 x {merged_df.shape[1]}列")

    print(f"\n  列名: {', '.join(merged_df.columns)}")

    print("\n  データサンプル:")
    print(merged_df.head())


# ============================================================================
# メイン処理関数
# ============================================================================


def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="SQLiteテーブルの結合を実行"
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLiteデータベースファイルパス"
    )
    parser.add_argument(
        "--tables",
        type=str,
        help="結合対象のテーブル名 (カンマ区切り)"
    )
    parser.add_argument(
        "--output-table",
        type=str,
        help="出力テーブル名 (未指定の場合は自動生成)"
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help="自動的に結合方法を決定する"
    )
    parser.add_argument(
        "--merge-strategy",
        type=str,
        choices=["vertical", "horizontal", "manual"],
        help="結合方法を指定 (vertical: 縦結合, horizontal: 横結合, manual: 手動結合)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="非対話モードで実行"
    )
    return parser.parse_args()


def select_tables_interactively(db_path: Path) -> list[str]:
    """対話的にテーブルを選択する"""
    available_tables = get_available_tables(db_path)

    if not available_tables:
        print("❌ 利用可能なテーブルがありません")
        return []

    print("\n📋 利用可能なテーブル:")
    for i, table in enumerate(available_tables, 1):
        print(f"  {i}. {table}")

    while True:
        choice = input("\n結合するテーブルの番号を入力してください (カンマ区切り): ").strip()
        if not choice:
            print("少なくとも1つのテーブルを選択してください")
            continue

        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            valid_indices = [i for i in indices if 0 <= i < len(available_tables)]

            if valid_indices:
                selected_tables = [available_tables[i] for i in valid_indices]
                print(f"✅ 選択されたテーブル: {', '.join(selected_tables)}")
                return selected_tables
            else:
                print("有効な番号を入力してください")
        except ValueError:
            print("有効な数字を入力してください")


def main():
    """メイン関数"""
    args = parse_arguments()

    print("🚀 テーブル結合ツール")
    print("=" * 60)

    # データベースパスの検証
    db_path = Path(args.db_file).expanduser().resolve()
    if not db_path.exists():
        print(f"❌ データベースファイルが見つかりません: {db_path}")
        return

    print(f"📁 データベース: {db_path}")

    # テーブル選択
    table_names = []
    if args.tables:
        table_names = [name.strip() for name in args.tables.split(',')]
    else:
        table_names = select_tables_interactively(db_path)
        if not table_names:
            return

    if len(table_names) < 2:
        print("❌ 結合には少なくとも2つのテーブルが必要です")
        return

    # テーブル情報の取得
    tables_info = get_tables_info(db_path, table_names)
    if len(tables_info) < 2:
        print("❌ 有効なテーブルが2つ未満です")
        return

    # テーブル情報の表示
    display_tables_info(tables_info)

    # 結合戦略の決定
    merge_strategy = args.merge_strategy
    if not merge_strategy:
        if args.auto_merge:
            merge_strategy = determine_merge_strategy(tables_info)
            if merge_strategy == "manual":
                merge_strategy = "vertical"  # デフォルト
        else:
            merge_strategy = get_merge_strategy_interactively(tables_info)

    if merge_strategy == "cancel":
        print("❌ 結合をキャンセルしました")
        return

    # データの読み込み
    dataframes = load_tables_data(db_path, table_names)

    # 結合の実行
    merged_df = None
    if merge_strategy == "vertical":
        merged_df = merge_vertically(dataframes, table_names)
    elif merge_strategy == "horizontal":
        merged_df = merge_horizontally(dataframes, table_names)
    elif merge_strategy == "manual":
        merged_df = merge_manually(dataframes, table_names, tables_info)

    if merged_df is None:
        print("❌ 結合に失敗しました")
        return

    # 出力テーブル名の決定
    output_table = args.output_table
    if not output_table:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_table = f"merged_{timestamp}"

    # 結果の保存と表示
    save_merged_table(db_path, merged_df, output_table)
    display_merge_results(merged_df, table_names, output_table)

    print("\n✅ テーブル結合が完了しました!")


if __name__ == "__main__":
    main()
