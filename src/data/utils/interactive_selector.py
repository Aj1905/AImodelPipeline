"""
対話的選択ユーティリティ。

ユーザーとの対話的なデータ選択機能を提供します。
"""

from pathlib import Path

from ..handlers.sqlite_handler import SQLiteHandler


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
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
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
        選択されたテーブル名、選択されなかった場合はNone
    """
    tables = get_available_tables(db_path)
    if not tables:
        print("利用可能なテーブルがありません")
        return None

    choice = _get_user_choice("利用可能なテーブル:", tables)
    return tables[choice - 1]


def get_table_columns(db_path: Path, table_name: str) -> list[tuple[str, str]]:
    """
    テーブルの列情報を取得する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名

    Returns:
        列名とデータ型のタプルのリスト
    """
    handler = SQLiteHandler(db_path)
    try:
        columns_info = handler.get_table_info(table_name)
        return [(col[1], col[2]) for col in columns_info]
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

    column_names = [col[0] for col in columns_info]
    column_types = [col[1] for col in columns_info]

    print("利用可能な列:")
    for i, (name, type_name) in enumerate(zip(column_names, column_types, strict=False), 1):
        print(f"  {i}. {name} ({type_name})")

    while True:
        try:
            choice = input("\n選択する列の番号を入力してください (複数の場合はカンマ区切り): ").strip()
            if not choice:
                print("少なくとも1つの列を選択してください")
                continue

            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            valid_indices = [i for i in indices if 0 <= i < len(column_names)]

            if valid_indices:
                selected_names = [column_names[i] for i in valid_indices]
                print(f"✅ 選択された列: {selected_names}")
                return selected_names
            else:
                print("有効な番号を入力してください")

        except ValueError:
            print("有効な数字を入力してください")


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
                    f"    {col_name}: {col_info['type']} (NULL: {col_info['null_count']}, ユニーク: {col_info['unique_count']})"
                )

    # ターゲット列と特徴量列を一度に選択
    print("\n🎯 ターゲット列と特徴量列を選択してください:")
    print("例: ターゲット列を7番、特徴量列を1,2,3,4,5,6番にする場合 → 7:1,2,3,4,5,6")

    columns_info = get_table_columns(db_path, table_name)
    column_names = [col[0] for col in columns_info]
    column_types = [col[1] for col in columns_info]

    print("\n利用可能な列:")
    for i, (name, type_name) in enumerate(zip(column_names, column_types, strict=False), 1):
        print(f"  {i}. {name} ({type_name})")

    while True:
        try:
            choice = input("\n選択してください (ターゲット列:特徴量列1,特徴量列2,...): ").strip()
            if not choice or ':' not in choice:
                print("正しい形式で入力してください (例: 7:1,2,3,4,5,6)")
                continue

            target_part, feature_part = choice.split(':', 1)

            # ターゲット列の処理
            target_index = int(target_part.strip()) - 1
            if not (0 <= target_index < len(column_names)):
                print("有効なターゲット列の番号を入力してください")
                continue

            target_column = column_names[target_index]

            # 特徴量列の処理
            if not feature_part.strip():
                print("少なくとも1つの特徴量列を選択してください")
                continue

            feature_indices = [int(x.strip()) - 1 for x in feature_part.split(',')]
            valid_feature_indices = [i for i in feature_indices if 0 <= i < len(column_names)]

            if not valid_feature_indices:
                print("有効な特徴量列の番号を入力してください")
                continue

            feature_columns = [column_names[i] for i in valid_feature_indices]

            # ターゲット列を特徴量から除外
            if target_column in feature_columns:
                feature_columns.remove(target_column)
                print(f"⚠️  ターゲット列 '{target_column}' を特徴量から除外しました")

            print(f"✅ 選択されたターゲット列: {target_column}")
            print(f"✅ 選択された特徴量列: {feature_columns}")

            return table_name, target_column, feature_columns

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
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        row_count = handler.fetch_one(count_query)[0]

        # 列情報を取得
        columns_info = handler.get_table_info(table_name)
        column_count = len(columns_info)

        # 各列の詳細情報を取得
        columns_detail = {}
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]

            try:
                # NULL値の数を取得
                null_query = f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL"
                null_count = handler.fetch_one(null_query)[0]

                # ユニーク値の数を取得
                unique_query = f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}"
                unique_count = handler.fetch_one(unique_query)[0]

                columns_detail[col_name] = {
                    "type": col_type,
                    "null_count": null_count,
                    "unique_count": unique_count
                }
            except Exception as e:
                columns_detail[col_name] = {
                    "error": str(e)
                }

        return {
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns_detail
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
    tables = get_available_tables(db_path)
    return table_name in tables


def validate_columns_exist(db_path: Path, table_name: str, column_names: list[str]) -> tuple[bool, list[str]]:
    """
    指定された列がテーブルに存在するかどうかを確認する

    Args:
        db_path: SQLiteデータベースファイルのパス
        table_name: テーブル名
        column_names: 確認する列名のリスト

    Returns:
        (すべての列が存在するかどうか, 存在しない列名のリスト)
    """
    columns_info = get_table_columns(db_path, table_name)
    existing_columns = [col[0] for col in columns_info]

    missing_columns = [col for col in column_names if col not in existing_columns]
    all_exist = len(missing_columns) == 0

    return all_exist, missing_columns
