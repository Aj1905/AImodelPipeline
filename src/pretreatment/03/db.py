import sqlite3

import pandas as pd

from .utils import _get_user_choice


def save_to_database(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """データフレームをSQLiteデータベースに保存する"""
    try:
        table_name = input("\n保存先のテーブル名を入力してください: ").strip()
        if not table_name:
            print("テーブル名が入力されていません。保存を中止します。")
            return
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            overwrite = _get_user_choice(f"テーブル '{table_name}' は既に存在します。", ["上書きする", "中止する"])
            if overwrite == 2:
                print("保存を中止します。")
                return
            cursor.execute(f"DROP TABLE `{table_name}`")
        df.to_sql(table_name, conn, index=False)
        print(f"✓ データをテーブル '{table_name}' に保存しました。")
        print(f"  行数: {df.shape[0]}, 列数: {df.shape[1]}")
    except Exception as e:
        print(f"保存中にエラーが発生しました: {e}")
        conn.rollback()
        raise
