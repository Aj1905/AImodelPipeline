"""
SQLiteデータベース操作のための再利用可能なユーティリティ関数

このモジュールは、SQLiteデータベースの操作に関する共通機能を提供します:
- テーブル一覧の取得
- 対話的なテーブル選択
- 列情報の取得
- 対話的な列選択
- データ読み込み
- ユーザー選択機能
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl

from src.data.sqlite_handler import SQLiteHandler

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def _get_user_choice(prompt: str, options: list[str]) -> int:
    """
    ユーザーから選択を取得する共通関数

    Args:
        prompt: 選択肢を表示するプロンプト
        options: 選択肢のリスト

    Returns:
        選択されたインデックス(1ベース)
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = input(f"\n選択してください (1-{len(options)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return choice_num
            else:
                print(f"1から{len(options)}の間で選択してください")
        except ValueError:
            print("有効な数字を入力してください")


def validate_db_path(db_path: Path) -> bool:
    """
    データベースパスの存在を確認する

    Args:
        db_path: データベースファイルのパス

    Returns:
        ファイルが存在する場合はTrue、そうでない場合はFalse
    """
    if not db_path.exists():
        print(f"❌ データベースファイルが見つかりません: {db_path}")
        return False
    return True


def load_data_from_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """
    指定されたテーブルからデータを読み込む(Pandas版)

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        読み込まれたデータのDataFrame
    """
    handler = SQLiteHandler(db_path)
    try:
        query = f"SELECT * FROM {table_name}"
        results = handler.fetch_all(query)

        if not results:
            print(f"テーブル '{table_name}' にデータがありません")
            return pd.DataFrame()

        # 列名を取得
        columns_info = handler.get_table_info(table_name)
        column_names = [col[1] for col in columns_info]

        # DataFrameを作成
        df = pd.DataFrame(results, columns=column_names)
        print(f"✓ データを読み込みました (形状: {df.shape})")
        return df

    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return pd.DataFrame()


def load_data_from_sqlite_polars(db_path: str, table: str) -> pl.DataFrame:
    """
    SQLiteから指定テーブルを読み込み、Polars DataFrameを返す

    Args:
        db_path: SQLiteデータベースファイルのパス
        table: テーブル名

    Returns:
        読み込まれたデータのPolars DataFrame
    """
    return pl.read_database_uri(query=f"SELECT * FROM {table}", uri=f"sqlite://{db_path}")


def interactive_setup(db_path: Path) -> tuple[str, str, list[str]]:
    """
    対話的にテーブル、ターゲット列、特徴量列を設定する

    Args:
        db_path: SQLiteデータベースファイルのパス

    Returns:
        (テーブル名, ターゲット列名, 特徴量列名のリスト)
    """
    print("🔍 データベース設定")
    print("=" * 40)

    # テーブル選択
    table_name = select_table_interactively(db_path)
    if not table_name:
        print("テーブルが選択されませんでした。処理を終了します。")
        raise ValueError("テーブルが選択されませんでした")

    print(f"✅ 選択されたテーブル: {table_name}")

    # テーブル情報の表示
    table_info = get_table_info_summary(db_path, table_name)
    if table_info:
        print("\n📊 テーブル情報:")
        print(f"  行数: {table_info['row_count']:,}")
        print(f"  列数: {table_info['column_count']}")
        print("  列の詳細:")
        for col_name, col_info in table_info["columns"].items():
            if "error" not in col_info:
                print(
                    f"    {col_name}: {col_info['type']} (NULL: {col_info['null_count']}, "
                    f"ユニーク: {col_info['unique_count']})"
                )

    # ターゲット列選択
    print("\n🎯 ターゲット列を選択してください:")
    target_column = select_columns_interactively(db_path, table_name)
    if not target_column:
        print("ターゲット列が選択されませんでした。処理を終了します。")
        raise ValueError("ターゲット列が選択されませんでした")
    target_column = target_column[0]  # 最初の選択をターゲットとする
    print(f"✅ 選択されたターゲット列: {target_column}")

    # 特徴量列選択
    print("\n🔧 特徴量列を選択してください:")
    feature_columns = select_columns_interactively(db_path, table_name)
    if not feature_columns:
        print("特徴量列が選択されませんでした。処理を終了します。")
        raise ValueError("特徴量列が選択されませんでした")

    # ターゲット列を特徴量から除外
    if target_column in feature_columns:
        feature_columns.remove(target_column)
        print(f"⚠️  ターゲット列 '{target_column}' を特徴量から除外しました")

    print(f"✅ 選択された特徴量列: {feature_columns}")

    return table_name, target_column, feature_columns


def get_available_tables(db_path: Path) -> list[str]:
    """
    データベースから利用可能なテーブル名のリストを取得する

    Args:
        db_path: SQLiteデータベースファイルのパス

    Returns:
        テーブル名のリスト
    """
    handler = SQLiteHandler(db_path)
    try:
        query = (
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        results = handler.fetch_all(query)
        return [row[0] for row in results]
    except Exception as e:
        print(f"テーブル一覧取得エラー: {e}")
        return []


def select_table_interactively(db_path: Path) -> str | None:
    """
    対話的にテーブルを選択する

    Args:
        db_path: SQLiteデータベースファイルのパス

    Returns:
        選択されたテーブル名、キャンセル時はNone
    """
    tables = get_available_tables(db_path)

    if not tables:
        print("利用可能なテーブルが見つかりませんでした")
        return None

    print("\n利用可能なテーブル:")
    for i, table in enumerate(tables, 1):
        # テーブルの行数を取得
        handler = SQLiteHandler(db_path)
        try:
            count_result = handler.fetch_one(f"SELECT COUNT(*) FROM {table}")
            row_count = count_result[0] if count_result else 0

            # テーブルの列情報を取得
            table_info = handler.get_table_info(table)
            column_count = len(table_info)

            print(f"  {i}. {table} (行数: {row_count}, 列数: {column_count})")
        except Exception as e:
            print(f"  {i}. {table} (情報取得エラー: {e})")

    while True:
        try:
            choice = input(
                f"\nテーブルを選択してください (1-{len(tables)}) または 'q' で終了: "
            ).strip()
            if choice.lower() == "q":
                return None

            table_index = int(choice) - 1
            if 0 <= table_index < len(tables):
                return tables[table_index]
            else:
                print(f"1から{len(tables)}の間で選択してください")
        except ValueError:
            print("有効な数字を入力してください")


def get_table_columns(db_path: Path, table_name: str) -> list[tuple[str, str]]:
    """
    テーブルの列名とデータ型を取得する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        列名とデータ型のタプルのリスト
    """
    handler = SQLiteHandler(db_path)
    try:
        return handler.get_table_info(table_name)
    except Exception as e:
        print(f"列情報取得エラー: {e}")
        return []


def select_columns_interactively(db_path: Path, table_name: str) -> list[str]:
    """
    対話的に列を選択する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        選択された列名のリスト
    """
    columns_info = get_table_columns(db_path, table_name)

    if not columns_info:
        print("列情報を取得できませんでした")
        return []

    print(f"\nテーブル '{table_name}' の列:")
    for i, (_cid, name, dtype, _notnull, _default_value, _pk) in enumerate(columns_info, 1):
        print(f"  {i}. {name} ({dtype})")

    while True:
        try:
            choice = input(
                f"\n対象の列番号をカンマ区切りで選択してください (1-{len(columns_info)}) または 'q' で終了: "
            ).strip()
            if choice.lower() == "q":
                return []

            selected_indices = [int(x.strip()) - 1 for x in choice.split(",")]
            if all(0 <= idx < len(columns_info) for idx in selected_indices):
                selected_columns = [columns_info[idx][1] for idx in selected_indices]
                return selected_columns
            else:
                print(f"1から{len(columns_info)}の間で選択してください")
        except ValueError:
            print("有効な数字を入力してください")


def get_table_info_summary(db_path: Path, table_name: str) -> dict:
    """
    テーブルの詳細情報を取得する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        テーブル情報の辞書
    """
    handler = SQLiteHandler(db_path)
    try:
        # 行数を取得
        count_result = handler.fetch_one(f"SELECT COUNT(*) FROM {table_name}")
        row_count = count_result[0] if count_result else 0

        # 列情報を取得
        columns_info = handler.get_table_info(table_name)
        column_count = len(columns_info)

        # 各列の基本統計を取得
        column_stats = {}
        for _cid, name, dtype, _notnull, _default_value, _pk in columns_info:
            try:
                # NULL値の数を取得
                null_count_result = handler.fetch_one(
                    f'SELECT COUNT(*) FROM {table_name} WHERE "{name}" IS NULL'
                )
                null_count = null_count_result[0] if null_count_result else 0

                # ユニーク値の数を取得
                unique_count_result = handler.fetch_one(
                    f'SELECT COUNT(DISTINCT "{name}") FROM {table_name}'
                )
                unique_count = unique_count_result[0] if unique_count_result else 0

                column_stats[name] = {
                    "type": dtype,
                    "null_count": null_count,
                    "unique_count": unique_count,
                    "null_ratio": null_count / row_count if row_count > 0 else 0,
                    "unique_ratio": unique_count / row_count if row_count > 0 else 0,
                }
            except Exception as e:
                column_stats[name] = {"type": dtype, "error": str(e)}

        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": column_count,
            "columns": column_stats,
        }
    except Exception as e:
        print(f"テーブル情報取得エラー: {e}")
        return {}


def validate_table_exists(db_path: Path, table_name: str) -> bool:
    """
    テーブルが存在するかどうかを確認する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        テーブルが存在する場合はTrue、そうでない場合はFalse
    """
    handler = SQLiteHandler(db_path)
    return handler.table_exists(table_name)


def validate_columns_exist(
    db_path: Path, table_name: str, column_names: list[str]
) -> tuple[bool, list[str]]:
    """
    指定された列がテーブルに存在するかどうかを確認する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名
        column_names: 確認する列名のリスト

    Returns:
        (全ての列が存在するかどうか, 存在しない列名のリスト)
    """
    handler = SQLiteHandler(db_path)
    try:
        table_columns = [col[1] for col in handler.get_table_info(table_name)]
        missing_columns = [col for col in column_names if col not in table_columns]
        return len(missing_columns) == 0, missing_columns
    except Exception as e:
        print(f"列存在確認エラー: {e}")
        return False, column_names


if __name__ == "__main__":
    """
    このファイルを直接実行した際のテスト用コード
    """
    print("🔧 SQLiteデータベース操作ユーティリティのテスト")
    print("=" * 50)

    # テスト用のデータベースパスを設定
    # 実際のデータベースファイルのパスに変更してください
    test_db_path = Path("data/database.sqlite")  # 実際のデータベースファイル

    print(f"テスト用データベースパス: {test_db_path}")

    # データベースファイルの存在確認
    if validate_db_path(test_db_path):
        print("✅ データベースファイルが見つかりました")

        # 利用可能なテーブルを取得
        tables = get_available_tables(test_db_path)
        if tables:
            print(f"📋 利用可能なテーブル: {tables}")

            # 最初のテーブルの情報を表示
            first_table = tables[0]
            table_info = get_table_info_summary(test_db_path, first_table)
            if table_info:
                print(f"\n📊 テーブル '{first_table}' の情報:")
                print(f"  行数: {table_info['row_count']:,}")
                print(f"  列数: {table_info['column_count']}")
        else:
            print("❌ 利用可能なテーブルが見つかりませんでした")
    else:
        print("❌ データベースファイルが見つかりませんでした")
        print("💡 実際のデータベースファイルのパスに変更してから再実行してください")

    print("\n✨ テスト完了")
