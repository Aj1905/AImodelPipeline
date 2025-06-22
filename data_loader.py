import sqlite3
import sys

import pandas as pd
import polars as pl

from .utils import _get_user_choice, execute_horizontal_concatenate, find_common_columns


def select_tables_interactively(db_path: str) -> list[str]:
    """複数のテーブルを選択させる"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()

    if not tables:
        print("テーブルが見つかりませんでした。")
        sys.exit(1)

    print("利用可能なテーブル:")
    table_info = {}
    for i, tbl in enumerate(tables, 1):
        conn = sqlite3.connect(db_path)
        df_info = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 1", conn)
        row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {tbl}", conn)
        conn.close()

        row_count_val = row_count.iloc[0]["count"]
        table_info[tbl] = row_count_val

        print(f"  {i}. {tbl} - 行数: {row_count_val}, 列数: {len(df_info.columns)}")

    sel = input("使用するテーブル番号をカンマ区切りで入力してください: ")
    try:
        idxs = [int(x.strip()) - 1 for x in sel.split(",")]
        chosen = [tables[i] for i in idxs if 0 <= i < len(tables)]
        if not chosen:
            raise ValueError

        if len(chosen) > 1:
            print("\n=== 選択されたテーブルの行数比較 ===")
            row_counts = [table_info[tbl] for tbl in chosen]
            min_rows = min(row_counts)
            max_rows = max(row_counts)

            for _i, tbl in enumerate(chosen):
                row_count = table_info[tbl]
                if row_count == min_rows:
                    status = " (最小)"
                elif row_count == max_rows:
                    status = " (最大)"
                else:
                    status = ""
                print(f"  {tbl}: {row_count}行{status}")

            if min_rows == max_rows:
                print("✓ 全てのテーブルの行数が同じです - HORIZONTAL_CONCATENATEが可能です")
            else:
                print(f"⚠️ 行数が異なります (最小: {min_rows}, 最大: {max_rows})")
                print("  - HORIZONTAL_CONCATENATEは使用できません")
                print("  - 他の結合方法(CROSS JOIN、CONCATENATE等)を検討してください")

        return chosen
    except Exception:
        print("無効な選択です。")
        sys.exit(1)


def _get_manual_join_keys(dfs: dict[str, pl.DataFrame], table1: str, table2: str) -> tuple[str, str]:
    """手動で結合キーを選択"""
    print(f"\n{table1} の列:")
    for j, col in enumerate(dfs[table1].columns, 1):
        print(f"  {j}. {col}")

    while True:
        try:
            key1_choice = int(input(f"{table1}の結合キー番号を入力: ").strip())
            if 1 <= key1_choice <= len(dfs[table1].columns):
                key1 = dfs[table1].columns[key1_choice - 1]
                break
            else:
                print(f"1~{len(dfs[table1].columns)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")

    print(f"\n{table2} の列:")
    for j, col in enumerate(dfs[table2].columns, 1):
        print(f"  {j}. {col}")

    while True:
        try:
            key2_choice = int(input(f"{table2}の結合キー番号を入力: ").strip())
            if 1 <= key2_choice <= len(dfs[table2].columns):
                key2 = dfs[table2].columns[key2_choice - 1]
                break
            else:
                print(f"1~{len(dfs[table2].columns)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")

    return key1, key2


def _handle_no_common_columns(dfs: dict[str, pl.DataFrame], table1: str, table2: str) -> str | tuple[str, str]:
    """共通カラムがない場合の処理"""
    print(f"⚠️ 警告: {table1} と {table2} の間に共通カラムがありません")

    join_options = [
        "CROSS JOIN - 全組み合わせで結合 (データ数が爆発的に増加する可能性)",
        "CONCATENATE - 行方向に連結 (同じ列構造が必要)",
        "HORIZONTAL_CONCATENATE - 行数が等しい場合に横に並べる",
        "SKIP - このテーブルをスキップ",
        "MANUAL - 手動で結合キーを指定",
    ]

    join_choice = _get_user_choice("結合方法を選択してください:", join_options)

    if join_choice == 1:
        return "CROSS_JOIN"
    elif join_choice == 2:
        return "CONCATENATE"
    elif join_choice == 3:
        return "HORIZONTAL_CONCATENATE"
    elif join_choice == 4:
        return "SKIP"
    elif join_choice == 5:
        return _get_manual_join_keys(dfs, table1, table2)
    else:
        return "SKIP"



def _load_tables_from_db(db_path: str, tables: list[str]) -> dict[str, pl.DataFrame]:
    """データベースからテーブルを読み込み"""
    conn = sqlite3.connect(db_path)
    dfs = {}

    for table in tables:
        df_pd = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        df_pd = df_pd.replace("", pd.NA).replace({"True": 1, "False": 0})
        dfs[table] = pl.from_pandas(df_pd)
        print(f"テーブル '{table}' 読み込み完了: {dfs[table].shape}")

    conn.close()
    return dfs


def _execute_cross_join(merged_df: pl.DataFrame, current_df: pl.DataFrame, current_table: str) -> pl.DataFrame:
    """CROSS JOINを実行"""
    print(f"CROSS JOIN実行: {current_table}")
    right_cols = current_df.columns
    rename_dict = {col: f"{current_table}_{col}" for col in right_cols}
    renamed_df = current_df.rename(rename_dict)
    result = merged_df.join(renamed_df, how="cross")
    print(f"→ CROSS JOIN後: {result.shape}")
    return result


def _execute_concatenate(
    merged_df: pl.DataFrame, current_df: pl.DataFrame, prev_table: str, current_table: str
) -> pl.DataFrame:
    """CONCATENATEを実行"""
    print(f"CONCATENATE実行: {prev_table} + {current_table}")
    if set(merged_df.columns) == set(current_df.columns):
        result = pl.concat([merged_df, current_df], how="vertical")
        print(f"→ CONCATENATE後: {result.shape}")
        return result
    else:
        print("❌ 列構造が異なるためCONCATENATEできません")
        print(f"  {prev_table}の列: {list(merged_df.columns)}")
        print(f"  {current_table}の列: {list(current_df.columns)}")
        return merged_df


def _execute_horizontal_concatenate(
    merged_df: pl.DataFrame, current_df: pl.DataFrame, prev_table: str, current_table: str
) -> pl.DataFrame:
    """HORIZONTAL_CONCATENATEを実行"""
    return execute_horizontal_concatenate(merged_df, current_df, prev_table, current_table)


def _execute_manual_join(
    merged_df: pl.DataFrame, current_df: pl.DataFrame, prev_table: str, current_table: str, key1: str, key2: str
) -> pl.DataFrame:
    """MANUAL結合を実行"""
    print(f"MANUAL結合実行: {prev_table}.{key1} = {current_table}.{key2}")
    right_cols = [col for col in current_df.columns if col != key2]
    rename_dict = {col: f"{current_table}_{col}" for col in right_cols}
    renamed_df = current_df.rename(rename_dict)
    renamed_df = renamed_df.rename({key2: key1})
    result = merged_df.join(renamed_df, on=key1, how="left")
    print(f"→ MANUAL結合後: {result.shape}")
    return result


def _execute_normal_join(
    merged_df: pl.DataFrame, current_df: pl.DataFrame, prev_table: str, current_table: str, join_key: str
) -> pl.DataFrame:
    """通常の結合を実行"""
    print(f"結合中: {prev_table}.{join_key} = {current_table}.{join_key}")
    right_cols = [col for col in current_df.columns if col != join_key]
    rename_dict = {col: f"{current_table}_{col}" for col in right_cols}
    renamed_df = current_df.rename(rename_dict)
    result = merged_df.join(renamed_df, on=join_key, how="left")
    print(f"→ 結合後: {result.shape}")
    return result


def _check_data_quality(merged_df: pl.DataFrame) -> None:
    """結合後のデータ品質をチェック"""
    print("\n=== 結合後のデータ品質チェック ===")
    merged_pd = merged_df.to_pandas()

    missing_info = merged_pd.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    if len(missing_cols) > 0:
        print(f"欠損値のある列 ({len(missing_cols)}列):")
        for col, count in missing_cols.items():
            rate = (count / len(merged_pd)) * 100
            print(f"  {col}: {count}個 ({rate:.1f}%)")
    else:
        print("✓ 欠損値はありません")

    print("\nデータ型分布:")
    dtype_counts = merged_pd.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}列")

    if len(merged_df.columns) > 100:
        print(f"\n⚠️ 警告: 列数が{len(merged_df.columns)}個と多いため、特徴量選択を慎重に行ってください")


def load_and_join_tables(db_path: str, tables: list[str]) -> pl.DataFrame:
    """複数テーブルを読み込み結合"""
    dfs = _load_tables_from_db(db_path, tables)

    if len(tables) == 1:
        return dfs[tables[0]]

    join_keys = find_common_columns(dfs)

    merged_df = dfs[tables[0]]
    print(f"\n開始テーブル: {tables[0]} ({merged_df.shape})")

    for i in range(1, len(tables)):
        current_table = tables[i]
        prev_table = tables[i - 1]
        join_config = join_keys.get((prev_table, current_table))

        if not join_config:
            print(f"⚠️ {prev_table} と {current_table} は結合設定がないためスキップします")
            continue

        if join_config == "SKIP":
            print(f"⚠️ {current_table} はスキップされます")
            continue
        elif join_config == "CROSS_JOIN":
            merged_df = _execute_cross_join(merged_df, dfs[current_table], current_table)
        elif join_config == "CONCATENATE":
            merged_df = _execute_concatenate(merged_df, dfs[current_table], prev_table, current_table)
        elif join_config == "HORIZONTAL_CONCATENATE":
            merged_df = _execute_horizontal_concatenate(merged_df, dfs[current_table], prev_table, current_table)
        elif isinstance(join_config, tuple):
            key1, key2 = join_config
            merged_df = _execute_manual_join(merged_df, dfs[current_table], prev_table, current_table, key1, key2)
        else:
            merged_df = _execute_normal_join(merged_df, dfs[current_table], prev_table, current_table, join_config)

    _check_data_quality(merged_df)
    return merged_df


def load_data_from_sqlite(db_path: str, table: str) -> pl.DataFrame:
    """SQLiteから指定テーブルを読み込み、Polars DataFrameを返す"""
    conn = sqlite3.connect(db_path)
    df_pd = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    df_pd = df_pd.replace("", pd.NA).replace({"True": 1, "False": 0})
    conn.close()
    return pl.from_pandas(df_pd)


def select_table_interactively(db_path: str) -> str:
    """データベース内テーブルを番号選択"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()
    if not tables:
        print("テーブルが見つかりませんでした。")
        sys.exit(1)
    print("利用可能なテーブル:")
    for i, tbl in enumerate(tables, 1):
        conn = sqlite3.connect(db_path)
        df_info = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 1", conn)
        row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {tbl}", conn)
        conn.close()

        print(f"  {i}. {tbl} - 行数: {row_count.iloc[0]['count']}, 列数: {len(df_info.columns)}")

    sel = input("使用するテーブル番号を入力してください: ")
    try:
        idx = int(sel) - 1
        return tables[idx]
    except Exception:
        print("無効な選択です。")
        sys.exit(1)
