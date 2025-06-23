"""
CSVの生データをそのままSQLiteに保存するスクリプト。

前処理なしでCSVファイルをSQLiteデータベースに変換します。
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.csv_reader import CSVReader
from src.data.sqlite_handler import SQLiteHandler


def parse_args():
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(description="CSVの生データをそのままSQLiteデータベースに保存します。")
    parser.add_argument("--csv-file", type=str, required=True, help="CSVファイルのパス")
    parser.add_argument(
        "--db-file",
        type=str,
        default="data/database.sqlite",
        help="SQLiteデータベースファイルのパス (デフォルト: data/database.sqlite)",
    )
    parser.add_argument("--table-name", type=str, required=True, help="保存先のテーブル名")
    parser.add_argument("--delimiter", type=str, default=",", help="CSVの区切り文字(デフォルト: ,)")
    parser.add_argument("--no-header", action="store_true", help="CSVファイルにヘッダーがない場合に指定")
    parser.add_argument(
        "--encoding", type=str, default="utf-8", help="CSVファイルのエンコーディング (デフォルト: utf-8)"
    )
    parser.add_argument("--headers", nargs="*", help="カラムヘッダー名を指定(ヘッダーがない場合のみ)")

    return parser.parse_args()


def infer_sqlite_type(value):
    """値からSQLiteの型を推測する。

    Args:
        value: 値

    Returns:
        str: SQLiteの型
    """
    if value is None or value == "":
        return "TEXT"

    # 整数型のチェック
    try:
        int(value)
        return "INTEGER"
    except (ValueError, TypeError):
        pass

    # 浮動小数点型のチェック
    try:
        float(value)
        return "REAL"
    except (ValueError, TypeError):
        pass

    # デフォルトは文字列型
    return "TEXT"


def infer_column_type(column_values):
    """列の値から最適なSQLiteの型を推測する。

    Args:
        column_values (list): 列の値のリスト

    Returns:
        str: 推測されたSQLiteの型
    """
    non_empty_values = [v for v in column_values if v is not None]

    if not non_empty_values:
        return "TEXT"

    # 型の推測(最初の非空値を使用)
    inferred_type = infer_sqlite_type(non_empty_values[0])

    # 整数として推測された場合、全ての値が整数かチェック
    if inferred_type == "INTEGER":
        for value in non_empty_values:
            try:
                int(value)
            except (ValueError, TypeError):
                inferred_type = "REAL"
                break

    # 実数として推測された場合、全ての値が数値かチェック
    if inferred_type == "REAL":
        for value in non_empty_values:
            try:
                float(value)
            except (ValueError, TypeError):
                inferred_type = "TEXT"
                break

    return inferred_type


def create_table_from_csv_data(sqlite_handler: SQLiteHandler, table_name: str, data) -> None:
    """CSVデータから自動的にテーブルを作成する。

    Args:
        sqlite_handler (SQLiteHandler): SQLiteハンドラー
        table_name (str): テーブル名
        data: CSVから読み込んだデータ

    Raises:
        ValueError: データが空の場合
    """
    if not data:
        raise ValueError("データが空です")

    # 最初の行からカラム名を取得
    first_row = data[0]
    columns = {}

    # 全行を見て最適な型を決定
    for column_name in first_row.keys():
        column_values = [row.get(column_name) for row in data]
        columns[column_name] = infer_column_type(column_values)

    # テーブル作成
    sqlite_handler.create_table(table_name, columns)
    print(f"テーブル '{table_name}' を作成しました:")
    for col, col_type in columns.items():
        print(f"  {col}: {col_type}")


def process_data(data, args):
    """CSVデータを処理し、必要に応じてヘッダーを適用する。

    Args:
        data: CSVから読み込んだデータ
        args: コマンドライン引数

    Returns:
        list: 処理済みのデータ
    """
    if not data:
        print("CSVファイルにデータがありません")
        return None

    # ヘッダーがない場合で、ヘッダー名が指定されている場合の処理
    if args.no_header and args.headers:
        # 最初の行のカラム数を取得
        first_row = data[0]
        actual_column_count = len(first_row.keys())
        provided_header_count = len(args.headers)

        # ヘッダー数とカラム数の一致チェック
        if provided_header_count != actual_column_count:
            print(
                f"エラー: 指定されたヘッダー数 ({provided_header_count}) とCSVのカラム数 ({actual_column_count}) が一致しません"
            )
            sys.exit(1)

        # ヘッダー名を適用
        renamed_data = []
        for row in data:
            new_row = {}
            for i, header_name in enumerate(args.headers):
                old_key = str(i)  # CSVReaderがno_headerでインデックスをキーにする
                if old_key in row:
                    new_row[header_name] = row[old_key]
            renamed_data.append(new_row)
        data = renamed_data
        print(f"ヘッダー名を適用しました: {args.headers}")

    # 空文字列をNoneに変換
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if value == "":
                cleaned_row[key] = None
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)

    return cleaned_data


def display_results(sqlite_handler, table_name, data_count, db_path):
    """処理結果を表示する。

    Args:
        sqlite_handler (SQLiteHandler): SQLiteハンドラー
        table_name (str): テーブル名
        data_count (int): 保存したデータの行数
        db_path (Path): データベースファイルのパス
    """
    print(f"完了: {data_count} 行のデータをテーブル '{table_name}' に保存しました")
    print(f"データベースファイル: {db_path}")

    # データの確認
    print("\n保存されたデータの確認:")
    sample_data = sqlite_handler.fetch_all(f"SELECT * FROM {table_name} LIMIT 5")
    if sample_data:
        print("最初の5行:")
        for i, row in enumerate(sample_data, 1):
            print(f"  {i}: {row}")

    # テーブル情報を表示
    table_info = sqlite_handler.get_table_info(table_name)
    print(f"\nテーブル情報 ({table_name}):")
    for column_info in table_info:
        print(f"  {column_info[1]}: {column_info[2]}")


def main():
    """メイン実行関数。

    CSVファイルの生データをそのままSQLiteデータベースに保存します。
    """
    args = parse_args()

    try:
        # CSVファイルを読み込み
        csv_reader = CSVReader(args.csv_file, encoding=args.encoding)
        has_header = not args.no_header

        print(f"CSVファイルを読み込み中: {args.csv_file}")
        data = csv_reader.read(delimiter=args.delimiter, has_header=has_header)

        # データの処理
        data = process_data(data, args)
        if data is None:
            return

        print(f"読み込み完了: {len(data)} 行のデータ")

        # データベースファイルのパスを解決
        db_path = Path(args.db_file)
        if not db_path.is_absolute():
            db_path = project_root / db_path

        # データベースディレクトリが存在しない場合は作成
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # SQLiteハンドラーを初期化
        sqlite_handler = SQLiteHandler(db_path)

        # テーブルが存在しない場合は作成
        if not sqlite_handler.table_exists(args.table_name):
            create_table_from_csv_data(sqlite_handler, args.table_name, data)
        else:
            print(f"テーブル '{args.table_name}' は既に存在します")

        # データを挿入
        print("データを挿入中...")
        sqlite_handler.insert_many(args.table_name, data)

        # 結果の表示
        display_results(sqlite_handler, args.table_name, len(data), db_path)

    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
