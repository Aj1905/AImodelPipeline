"""
SQLiteデータベースハンドラー。

SQLiteデータベースの操作を抽象化するクラスを提供します。
"""

import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple


class SQLiteHandler:
    """SQLiteデータベースの操作を管理するクラス。"""

    def __init__(self, db_path: Path):
        """SQLiteHandlerを初期化する。

        Args:
            db_path (Path): SQLiteデータベースファイルのパス
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント。"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーのエグジットポイント。"""
        self.close()

    def connect(self) -> None:
        """データベースに接続する。"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """データベース接続を閉じる。"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """SQLクエリを実行する。

        Args:
            query (str): 実行するSQLクエリ
            params (tuple, optional): クエリパラメータ. デフォルトは ().

        Returns:
            sqlite3.Cursor: 実行結果のカーソル
        """
        if self.connection is None:
            self.connect()
        
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Tuple]:
        """単一のレコードを取得する。

        Args:
            query (str): 実行するSQLクエリ
            params (tuple, optional): クエリパラメータ. デフォルトは ().

        Returns:
            Optional[Tuple]: 取得したレコード、存在しない場合はNone
        """
        cursor = self.execute(query, params)
        return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = ()) -> List[Tuple]:
        """すべてのレコードを取得する。

        Args:
            query (str): 実行するSQLクエリ
            params (tuple, optional): クエリパラメータ. デフォルトは ().

        Returns:
            List[Tuple]: 取得したレコードのリスト
        """
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def commit(self) -> None:
        """変更をコミットする。"""
        if self.connection:
            self.connection.commit()

    def rollback(self) -> None:
        """変更をロールバックする。"""
        if self.connection:
            self.connection.rollback()

    def table_exists(self, table_name: str) -> bool:
        """テーブルが存在するかどうかを確認する。

        Args:
            table_name (str): 確認するテーブル名

        Returns:
            bool: テーブルが存在する場合はTrue、そうでない場合はFalse
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table_name,))
        return result is not None

    def get_table_info(self, table_name: str) -> List[Tuple]:
        """テーブルの情報を取得する。

        Args:
            table_name (str): テーブル名

        Returns:
            List[Tuple]: テーブル情報のリスト
        """
        query = f"PRAGMA table_info({table_name})"
        return self.fetch_all(query)

    def create_table(self, table_name: str, columns: List[Tuple[str, str]]) -> None:
        """テーブルを作成する。

        Args:
            table_name (str): 作成するテーブル名
            columns (List[Tuple[str, str]]): 列名とデータ型のタプルのリスト
        """
        column_definitions = ", ".join([f"{col[0]} {col[1]}" for col in columns])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        self.execute(query)
        self.commit()

    def insert_many(self, table_name: str, data: List[Tuple]) -> None:
        """複数のレコードを挿入する。

        Args:
            table_name (str): 挿入先のテーブル名
            data (List[Tuple]): 挿入するデータのリスト
        """
        if not data:
            return

        placeholders = ", ".join(["?" for _ in data[0]])
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        
        cursor = self.connection.cursor()
        cursor.executemany(query, data)
        self.commit()

    def insert_one(self, table_name: str, data: Tuple) -> None:
        """単一のレコードを挿入する。

        Args:
            table_name (str): 挿入先のテーブル名
            data (Tuple): 挿入するデータ
        """
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        
        self.execute(query, data)
        self.commit()

    def update(self, table_name: str, set_values: dict, where_condition: str, where_params: tuple = ()) -> None:
        """レコードを更新する。

        Args:
            table_name (str): 更新対象のテーブル名
            set_values (dict): 更新する値の辞書
            where_condition (str): WHERE条件
            where_params (tuple, optional): WHERE条件のパラメータ. デフォルトは ().
        """
        set_clause = ", ".join([f"{key} = ?" for key in set_values.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_condition}"
        
        params = tuple(set_values.values()) + where_params
        self.execute(query, params)
        self.commit()

    def delete(self, table_name: str, where_condition: str, where_params: tuple = ()) -> None:
        """レコードを削除する。

        Args:
            table_name (str): 削除対象のテーブル名
            where_condition (str): WHERE条件
            where_params (tuple, optional): WHERE条件のパラメータ. デフォルトは ().
        """
        query = f"DELETE FROM {table_name} WHERE {where_condition}"
        self.execute(query, where_params)
        self.commit()

    def get_table_count(self, table_name: str) -> int:
        """テーブルのレコード数を取得する。

        Args:
            table_name (str): テーブル名

        Returns:
            int: レコード数
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.fetch_one(query)
        return result[0] if result else 0

    def get_table_schema(self, table_name: str) -> List[Tuple]:
        """テーブルのスキーマを取得する。

        Args:
            table_name (str): テーブル名

        Returns:
            List[Tuple]: スキーマ情報のリスト
        """
        query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table_name,))
        return result[0] if result else None 