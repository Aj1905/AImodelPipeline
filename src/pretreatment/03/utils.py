import pandas as pd


def check_division_by_zero(operation: str, col2_num: pd.Series) -> None:
    """ゼロ除算の警告を表示する"""
    if operation == "/" and (col2_num == 0).any():
        zero_count = (col2_num == 0).sum()
        print(f"警告: 列2に {zero_count} 個のゼロ値があります。")
        print("ゼロ除算によりNaNが発生します。")


def validate_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """指定された列がDataFrameに存在するかチェックする"""
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"エラー: 以下の列が存在しません: {missing_columns}")
        print(f"利用可能な列: {list(df.columns)}")
        return False
    return True


def _load_data(conn, table_name: str) -> pd.DataFrame:
    """指定テーブルのデータを読み込む"""
    return pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)


def _get_user_choice(prompt: str, options: list) -> int:
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input("選択: ").strip())
            if 1 <= choice <= len(options):
                return choice
            print(f"無効な選択です。1~{len(options)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_column_choice(df: pd.DataFrame, prompt: str, allow_multiple: bool = False, allow_all: bool = False) -> list:
    print(f"\n{prompt}")
    print("現在の列一覧:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    if allow_all:
        print(f"{len(df.columns) + 1}. すべての列")
    if allow_multiple:
        print("複数選択の場合はカンマ区切りで入力(例: 1,3,5)")
    while True:
        try:
            user_input = input("選択: ").strip()
            if not user_input:
                return []
            if allow_all and user_input == str(len(df.columns) + 1):
                return list(df.columns)
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            if all(0 <= idx < len(df.columns) for idx in indices):
                return [df.columns[idx] for idx in indices]
            print(f"無効な番号です。1~{len(df.columns)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")
