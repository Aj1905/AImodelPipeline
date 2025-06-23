"""
データ読み込みユーティリティ。

データベースやファイルからのデータ読み込み機能を提供します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import polars as pl

from ..handlers.sqlite_handler import SQLiteHandler


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