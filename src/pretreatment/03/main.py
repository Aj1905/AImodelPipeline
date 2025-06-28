import sqlite3

import pandas as pd

from .arithmetic import arithmetic_columns
from .columns import delete_columns
from .date_ops import (
    add_daily_sum_column,
    add_holiday_flag_column,
    add_monthly_sum_column,
    add_weekday_column,
    add_year_month_day,
    convert_date_format,
)
from .db import save_to_database
from .overview import overview_data
from .reference import generate_column_from_reference
from .rows import delete_rows
from .text import replace_text
from .type_conversion import convert_column_types
from .utils import _get_user_choice, _load_data
from .visualization import (
    basic_statistics,
    plot_3d_histogram,
    plot_counter_all,
    plot_scatter,
)

DB_PATH = "/Users/aj/Documents/forecasting_poc/data/database.sqlite"

def _get_available_tables(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def _display_tables(tables: list[str], conn: sqlite3.Connection) -> None:
    print("\n利用可能なテーブル一覧:")
    for i, table in enumerate(tables, 1):
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM `{table}`", conn)["count"][0]
        print(f"{i}. {table} (行数: {count})")


def _execute_function(func, df: pd.DataFrame) -> pd.DataFrame:
    if func in [plot_counter_all, basic_statistics]:
        func(df)
        return df
    if func == plot_scatter:
        x = df.columns[_get_user_choice("X軸の列を選択してください:", list(df.columns)) - 1]
        y = df.columns[_get_user_choice("Y軸の列を選択してください:", list(df.columns)) - 1]
        func(df, x, y)
        return df
    if func == plot_3d_histogram:
        x = df.columns[_get_user_choice("X軸の列を選択してください:", list(df.columns)) - 1]
        y = df.columns[_get_user_choice("Y軸の列を選択してください:", list(df.columns)) - 1]
        bins = int(input("ビン数(デフォルト10): ") or 10)
        func(df, x, y, bins)
        return df
    result = func(df)
    return result if isinstance(result, pd.DataFrame) else df


def run_pretreatment() -> None:
    print("\n" + "=" * 50)
    print("データ前処理ツール - SQLite保存機能付き")
    print("=" * 50)
    print(f"データベースパス: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        tables = _get_available_tables(conn)
        if not tables:
            print("データベースにテーブルが存在しません")
            return
        _display_tables(tables, conn)
        table_choice = _get_user_choice("読み込むテーブルを選択してください:", tables)
        selected_table = tables[table_choice - 1]
        df = _load_data(conn, selected_table)
        print(f"\nテーブル '{selected_table}' を読み込みました (行: {df.shape[0]}, 列: {df.shape[1]})")
        available_funcs = [
            overview_data,
            delete_rows,
            delete_columns,
            replace_text,
            arithmetic_columns,
            generate_column_from_reference,
            convert_date_format,
            add_weekday_column,
            add_holiday_flag_column,
            add_year_month_day,
            add_daily_sum_column,
            add_monthly_sum_column,
            plot_counter_all,
            basic_statistics,
            plot_scatter,
            plot_3d_histogram,
            convert_column_types,
        ]
        func_names = [
            "overview_data - データ概要表示",
            "delete_rows - 行削除",
            "delete_columns - 列削除",
            "replace_text - テキスト置換",
            "arithmetic_columns - 列の四則演算",
            "generate_column_from_reference - 参照列からの新列生成",
            "convert_date_format - 日付形式変換",
            "add_weekday_column - 曜日列追加",
            "add_holiday_flag_column - 祝日フラグ追加",
            "add_year_month_day - 年・月・日列追加",
            "add_daily_sum_column - 日付ごとに合計",
            "add_monthly_sum_column - 月ごとに合計",
            "plot_counter_all - 全列のカウントプロット",
            "basic_statistics - 基本統計量表示",
            "plot_scatter - 散布図",
            "plot_3d_histogram - 3次元ヒストグラム",
            "convert_column_types - 列のデータ型変換",
        ]
        while True:
            func_choices = [*func_names, "保存して終了", "保存せずに終了"]
            choice = _get_user_choice("適用する前処理関数を選択してください:", func_choices)
            if choice == len(func_choices) - 1:
                save_to_database(df, conn)
                break
            if choice == len(func_choices):
                if _get_user_choice("変更を保存せずに終了しますか?", ["はい", "いいえ"]) == 1:
                    break
                continue
            func = available_funcs[choice - 1]
            print(f"\n選択した関数: {func.__name__}")
            try:
                df = _execute_function(func, df)
                print(f"\n処理完了: {func.__name__}")
                print(f"現在のデータ形状: {df.shape}")
                if _get_user_choice("この時点でデータを保存しますか?", ["いいえ", "はい"]) == 2:
                    save_to_database(df, conn)
            except Exception as e:
                print(f"関数実行中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("プログラムを終了します")


if __name__ == "__main__":
    run_pretreatment()
