import math
import sqlite3

import jpholiday
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# SQLite DBファイルのパスをここに設定してください
DB_PATH = "/Users/aj/Documents/AImodelPipeline/data/database.sqlite"


# 必要な関数を直接定義
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
    """
    指定したテーブル名のデータを読み込み、DataFrameで返します。
    複数テーブルを読み込む場合はこの関数を繰り返し呼び出してください。
    """
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
            else:
                print(f"無効な選択です。1~{len(options)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_column_choice(
    df: pd.DataFrame, prompt: str, allow_multiple: bool = False, allow_all: bool = False
) -> list:
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
            else:
                print(f"無効な番号です。1~{len(df.columns)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


# --- ここから汎用処理関数群 ---

### 1. データの概要


def overview_data(df: pd.DataFrame) -> None:
    def _get_column_types(df):
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_cols = [
            col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        missing_cols = df.columns[df.isna().any()].tolist()
        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "missing": missing_cols
        }

    col_types = _get_column_types(df)

    print("=== データ概要 ===")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"数値列: {len(col_types['numeric'])}")
    print(f"カテゴリ列: {len(col_types['categorical'])}")
    print(f"欠損値あり列: {len(col_types['missing'])}")

    # 列選択の入力受付
    cols_input = input(
        "表示する列をカンマ区切りで入力 (例: col1,col2)\n"
        "全て表示: all, 終了: exit\n"
        "入力: "
    ).strip()

    if cols_input.lower() == "exit":
        return

    if cols_input.lower() == "all":
        print("\n全列を表示:")
        print(df.head())
        return

    selected_cols = [col.strip() for col in cols_input.split(",") if col.strip()]

    # 有効な列のフィルタリング
    valid_cols = [col for col in selected_cols if col in df.columns]
    invalid_cols = set(selected_cols) - set(valid_cols)

    if invalid_cols:
        print(f"次の列は存在しません: {', '.join(invalid_cols)}")

    if not valid_cols:
        print("有効な列が選択されていません")
        return

    print(f"\n選択列 ({', '.join(valid_cols)}) の先頭5行:")
    print(df[valid_cols].head())

    # 続けて選択するか確認
    cont = input("\n他の列を表示しますか? (y/n): ").strip().lower()
    if cont != "y":
        return


### 2. 行削除


def _handle_condition_deletion(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """条件に基づく行削除を実行"""
    try:
        # 条件式を評価
        mask = df.eval(condition)
        deleted_count = mask.sum()
        if deleted_count > 0:
            df = df[~mask]
            print(f"✓ 条件 '{condition}' で {deleted_count} 行を削除しました")
        else:
            print(f"条件 '{condition}' に該当する行はありませんでした")
    except Exception as e:
        print(f"❌ 条件式の評価中にエラーが発生しました: {e}")
    return df


def _handle_column_deletion(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """指定列の削除を実行"""
    existing_columns = [col for col in columns if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"✓ 列 {existing_columns} を削除しました")
    else:
        print("指定された列は存在しませんでした")
    return df


def _handle_duplicate_deletion(
    df: pd.DataFrame, subset: list[str] | None
) -> pd.DataFrame:
    """重複行の削除を実行"""
    original_count = len(df)
    if subset:
        df = df.drop_duplicates(subset=subset)
    else:
        df = df.drop_duplicates()
    deleted_count = original_count - len(df)
    if deleted_count > 0:
        print(f"✓ 重複行 {deleted_count} 行を削除しました")
    else:
        print("重複行はありませんでした")
    return df


def delete_rows(
    df: pd.DataFrame,
    condition: str | None = None,
    columns: list[str] | None = None,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    if condition:
        df = _handle_condition_deletion(df, condition)
    if columns:
        df = _handle_column_deletion(df, columns)
    if subset is not None:
        df = _handle_duplicate_deletion(df, subset)
    return df


### 3. 列削除


def delete_columns(df: pd.DataFrame, columns: list | str | None = None) -> pd.DataFrame:
    """
    DataFrameから指定した列を削除する
    """
    if columns is None:
        print("\n📋 現在の列一覧:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        # モード選択
        print("\n削除方法を選択:")
        print("  1. 削除する列を指定")
        print("  2. 残す列を指定")

        while True:
            mode_input = input("選択 (1 or 2): ").strip()
            if mode_input in ["1", "2"]:
                is_delete_mode = mode_input == "1"
                break
            print("❌ 1 または 2 を入力してください")

        # 列番号の入力
        action = "削除" if is_delete_mode else "残す"
        while True:
            user_input = input(f"\n{action}列番号を入力 (例: 1,3,5): ").strip()

            if not user_input:
                print("処理を中止します。")
                return df

            try:
                # 入力を解析して列名を取得
                indices = [int(x.strip()) - 1 for x in user_input.split(",")]

                # 範囲チェック
                if any(i < 0 or i >= len(df.columns) for i in indices):
                    raise IndexError

                selected_cols = [df.columns[i] for i in indices]
                break

            except (ValueError, IndexError):
                print(f"❌ 無効な入力です。1~{len(df.columns)}の範囲で入力してください")

        cols_to_drop = (
            selected_cols if is_delete_mode
            else [col for col in df.columns if col not in selected_cols]
        )
    else:
        # 文字列の場合はリストに変換
        if isinstance(columns, str):
            columns = [columns]
        cols_to_drop = columns

    # 削除実行
    result = df.drop(columns=cols_to_drop)
    print(f"\n✅ 削除完了: {cols_to_drop}")
    print(f"📊 新しい形状: {result.shape}")

    return result


def replace_text(
    df: pd.DataFrame,
    column: str | None = None,
    to_replace: str | None = None,
    value: str | None = None
) -> pd.DataFrame:
    if to_replace is None:
        to_replace = input("検索文字列を入力してください: ").strip()

    if column is None:
        cols_with_all = [*list(df.columns), "すべての列"]
        col_choice = _get_user_choice("置換対象の列を選択してください:", cols_with_all)

        if col_choice <= len(df.columns):
            column = df.columns[col_choice - 1]
        else:
            column = None

    if value is None:
        value = input("置換後の文字列を入力してください: ").strip()

    result = df.copy()
    # 対象列の設定
    if column:
        if column not in result.columns:
            print(f"警告: 列 '{column}' が存在しません。")
            print(f"利用可能列: {list(result.columns)}")
            return result
        target_cols = [column]
    else:
        target_cols = result.columns.tolist()

    # 置換処理
    total_replaced = 0
    for col in target_cols:
        before = result[col].astype(str)
        after = before.str.replace(to_replace, value, regex=False)
        count = (before != after).sum()
        total_replaced += count
        result[col] = after

    print(f"置換完了: '{to_replace}' → '{value}' | 合計置換数: {total_replaced}")
    return result


# ---arithmetic_columns sub functions---


def _get_column_names(df, col1, col2):
    """列名を取得または選択する"""
    if col1 is None:
        col1_choice = _get_user_choice(
            "計算に使用する列1を選択してください:",
            list(df.columns)
        )
        col1 = df.columns[col1_choice - 1]

    if col2 is None:
        col2_choice = _get_user_choice(
            "計算に使用する列2を選択してください:",
            list(df.columns)
        )
        col2 = df.columns[col2_choice - 1]

    return col1, col2


def _validate_columns(df, col1, col2):
    """列の存在チェック"""
    return validate_columns(df, [col1, col2])


def _get_operation(operation):
    """演算子を取得または選択する"""
    if operation is None:
        operations = ["+", "-", "*", "/"]
        op_choice = _get_user_choice("演算を選択してください:", operations)
        operation = operations[op_choice - 1]
    return operation


def _get_new_column_name(df: pd.DataFrame, new_col: str | None = None) -> str:
    """新しい列名を取得します"""
    if new_col is None:
        new_col = input("生成する新列名を入力してください: ").strip()
        if not new_col:
            raise ValueError("新列名が入力されていません。")

    if new_col in df.columns:
        overwrite_choice = _get_user_choice(
            f"列 '{new_col}' は既に存在します。",
            ["上書きする", "処理を中止"]
        )
        if overwrite_choice == 2:
            raise ValueError("処理を中止します。")

    return new_col


def _prepare_numeric_data(df, col1, col2):
    """数値データに変換し、警告を表示する"""
    col1_num = pd.to_numeric(df[col1], errors="coerce")
    col2_num = pd.to_numeric(df[col2], errors="coerce")

    # NaN警告
    for name, col_data in [(col1, col1_num), (col2, col2_num)]:
        nans = col_data.isna().sum()
        if nans > 0:
            print(f"警告: 列 '{name}' に数値変換できない値が {nans} 個あります")

    return col1_num, col2_num


def _check_division_by_zero(operation, col2_num):
    """ゼロ除算の警告"""
    check_division_by_zero(operation, col2_num)


def _perform_calculation(col1_num, col2_num, operation):
    """計算を実行する"""
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y
    }
    return operations[operation](col1_num, col2_num)


def _show_statistics(result, col1, col2, operation, new_col):
    """統計情報を表示する"""
    stats = result[new_col].describe()
    print(f"★ {col1} {operation} {col2} を '{new_col}' 列に追加しました。")
    print(f"  count={stats['count']:.0f}, mean={stats['mean']:.2f}, ")
    print(f"  min={stats['min']:.2f}, max={stats['max']:.2f}")

    nan_cnt = result[new_col].isna().sum()
    if nan_cnt > 0:
        print(f"  NaN (計算不能) が {nan_cnt} 個あります")


# ---arithmetic_columns sub functions---


def arithmetic_columns(
    df: pd.DataFrame,
    col1: str | None = None,
    col2: str | None = None,
    operation: str | None = None,
    new_col: str | None = None,
) -> pd.DataFrame:
    result = df.copy()

    # 列名取得
    col1, col2 = _get_column_names(df, col1, col2)

    # 列存在チェック
    if not _validate_columns(result, col1, col2):
        return df

    # 演算子取得
    operation = _get_operation(operation)

    # 新列名取得
    new_col = _get_new_column_name(result, new_col)
    if new_col is None:
        return df

    # 数値データ準備
    col1_num, col2_num = _prepare_numeric_data(result, col1, col2)

    # ゼロ除算チェック
    _check_division_by_zero(operation, col2_num)

    # 計算実行
    result[new_col] = _perform_calculation(col1_num, col2_num, operation)

    # 統計表示
    _show_statistics(result, col1, col2, operation, new_col)

    return result


# ---generate_column_from_reference sub functions---


def _get_mapping_from_user(
    search_list: list[str] | None = None, replace_list: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """ユーザーからマッピング情報を取得します"""
    if search_list is None:
        search_input = input("変換前の文字列をカンマ区切りで入力してください: ").strip()
        if not search_input:
            raise ValueError("変換前リストが入力されていません。")
        search_list = [s.strip() for s in search_input.split(",")]

    if replace_list is None:
        replace_input = input(
            "変換後の文字列をカンマ区切りで入力してください: "
        ).strip()
        if not replace_input:
            raise ValueError("変換後リストが入力されていません。")
        replace_list = [s.strip() for s in replace_input.split(",")]

    if len(search_list) != len(replace_list):
        raise ValueError(
            f"変換前({len(search_list)})と変換後({len(replace_list)})の"
            f"要素数が一致しません"
        )

    return search_list, replace_list


def _apply_mapping(
    df: pd.DataFrame,
    ref_col: str,
    mapping: dict[str, str],
    new_col: str
) -> pd.DataFrame:
    """マッピングを適用して新しい列を追加します"""
    result = df.copy()
    result[new_col] = result[ref_col].astype(str).map(mapping)

    # 未マッピング値の処理
    unmapped = [v for v in result[ref_col].astype(str).unique() if v not in mapping]
    if unmapped:
        print(f"マッピングにない値: {unmapped}")
        dv = input("未マッピング値のデフォルトを入力(空白でNaN): ").strip()
        if dv:
            result[new_col] = result[new_col].fillna(dv)

    # 列順調整
    cols = list(result.columns)
    idx = cols.index(ref_col) + 1
    cols.remove(new_col)
    cols.insert(idx, new_col)
    result = result[cols]

    return result


# ---generate_column_from_reference sub functions---


def generate_column_from_reference(
    df: pd.DataFrame,
    ref_col: str | None = None,
    search_list: list[str] | None = None,
    replace_list: list[str] | None = None,
    new_col: str | None = None,
) -> pd.DataFrame:
    """
    参照列の値をもとにマッピングした新列を追加します。
    引数未指定時は対話式で入力を促します。
    """
    try:
        # 参照列名の取得
        if ref_col is None:
            ref_col_choice = _get_user_choice(
                "参照する列を選択してください:",
                list(df.columns)
            )
            ref_col = df.columns[ref_col_choice - 1]

        if ref_col not in df.columns:
            raise ValueError(
                f"列 '{ref_col}' が存在しません。"
                f"利用可能列: {list(df.columns)}"
            )

        # マッピング情報の取得
        search_list, replace_list = _get_mapping_from_user(search_list, replace_list)
        mapping = dict(zip(search_list, replace_list, strict=False))
        print(f"マッピング: {mapping}")

        # 新列名の取得
        new_col = _get_new_column_name(df, new_col)

        # マッピングの適用
        result = _apply_mapping(df, ref_col, mapping, new_col)

        # 結果サマリ
        cnt_s = result[new_col].notna().sum()
        cnt_nan = result[new_col].isna().sum()
        print(f"★ 新列 '{new_col}' を追加しました。")
        print(f"  成功: {cnt_s}, NaN: {cnt_nan}, shape: {result.shape}")

        return result

    except ValueError as e:
        print(f"エラー: {e!s}")
        return df


def convert_date_format(
    df: pd.DataFrame, column_name: str | None = None, input_format: str | None = None
) -> pd.DataFrame:
    """
    日付データを指定形式に変換する対話式関数。
    - column_name または input_format が未指定の場合、プロンプトで入力を促します。
    """
    result = df.copy()

    # 列名入力
    if column_name is None:
        col_choice = _get_user_choice(
            "日付データの列を選択してください:",
            list(df.columns)
        )
        column_name = df.columns[col_choice - 1]

    if column_name not in result.columns:
        print(f"エラー: 列 '{column_name}' が存在しません")
        return df

    # フォーマット入力
    input(
        "日付フォーマットを入力 (例: %Y-%m-%d、未入力で自動判定): "
    ).strip() or None

    try:
        result[column_name] = pd.to_datetime(
            result[column_name],
            format="%Y-%m-%d",
            errors="coerce"
        )
        print(f"✓ 日付変換完了: {column_name} 列 ({input_format} → %Y-%m-%d)")
    except Exception as e:
        print(f"✗ 変換エラー: {e}")
        print("入力形式パターンを確認してください")
        return df

    return result


def add_weekday_column(
    df: pd.DataFrame, date_column: str | None = None, weekday_column_name: str = "曜日"
) -> pd.DataFrame:
    """
    対話式で日付列を選択し、対応する曜日列を追加します。
    - date_column または weekday_column_name が未指定の場合、プロンプトで取得します。
    """
    result = df.copy()

    # 日付列選択
    if date_column is None:
        print(f"\nデータ行数: {len(result)}, 列数: {len(result.columns)}")
        col_choice = _get_user_choice(
            "日付列を選択してください(%Y-%m-%d形式):",
            list(result.columns)
        )
        date_column = result.columns[col_choice - 1]

    if date_column not in result.columns:
        print(f"エラー: 列 '{date_column}' が存在しません")
        return df

    # weekday_column_name入力
    if weekday_column_name is None or weekday_column_name == "曜日":
        name = input(
            "追加する曜日列の名前を入力してください (デフォルト: '曜日'): "
        ).strip()
        if name:
            weekday_column_name = name

    # 型チェック
    try:
        pd.to_datetime(result[date_column].dropna())
    except Exception as e:
        print(f"⚠ 警告: '{date_column}' 列の値が日付として認識できません: {e}")
        continue_choice = _get_user_choice(
            "処理を続行しますか?",
            ["続行する", "中止する"]
        )
        if continue_choice == 2:
            return df

    # 曜日の追加
    result[weekday_column_name] = pd.to_datetime(result[date_column]).dt.day_name()
    print(f"✓ 曜日列 '{weekday_column_name}' を追加しました。")

    # プレビュー表示
    preview = result[[date_column, weekday_column_name]].head()
    print("結果プレビュー:")
    print(preview)

    return result


def add_holiday_flag_column(
    df: pd.DataFrame,
    date_column: str | None = None,
    holiday_column_name: str = "祝日フラグ"
) -> pd.DataFrame:
    result = df.copy()

    if date_column is None:
        print(f"データ行数: {len(result)}, 列数: {len(result.columns)}")
        col_choice = _get_user_choice("日付列を選択してください:", list(result.columns))
        date_column = result.columns[col_choice - 1]

    if date_column not in result.columns:
        print(f"エラー: 列 '{date_column}' が存在しません")
        return df

    if holiday_column_name == "祝日フラグ":
        name = input(
            "追加する祝日フラグ列の名前を入力してください (デフォルト: '祝日フラグ'): "
        ).strip()
        if name:
            holiday_column_name = name

    try:
        pd.to_datetime(result[date_column].dropna())
    except Exception as e:
        print(f"⚠ 警告: '{date_column}' 列の値が日付として認識できません: {e}")
        choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if choice == 2:
            return df

    dates = pd.to_datetime(result[date_column])
    result[holiday_column_name] = dates.apply(
        lambda d: bool(d and jpholiday.is_holiday(d))
    )

    print(f"✓ 祝日フラグ列 '{holiday_column_name}' を追加しました。")
    preview = result[[date_column, holiday_column_name]].head()
    print("結果プレビュー:")
    print(preview)

    return result


def add_year_month_day(
    df: pd.DataFrame,
    date_column: str | None = None,
    remove_original: bool | None = None
) -> pd.DataFrame:
    result = df.copy()

    # 日付列選択
    if date_column is None:
        print(f"\nデータ行数: {len(result)}, 列数: {len(result.columns)}")
        col_choice = _get_user_choice(
            "日付列を選択してください(%Y-%m-%d形式):",
            list(result.columns)
        )
        date_column = result.columns[col_choice - 1]

    if date_column not in result.columns:
        print(f"エラー: 列 '{date_column}' が存在しません")
        return df

    # 日付列をdatetime型に変換
    try:
        result[date_column] = pd.to_datetime(
            result[date_column],
            format="%Y-%m-%d",
            errors="coerce"
        )
    except Exception as e:
        print(f"日付変換エラー: {e}")
        return df

    # 変換エラーがあった場合の警告
    invalid_count = result[date_column].isna().sum()
    if invalid_count > 0:
        print(f"警告: {invalid_count}件の無効な日付形式をNaNに変換しました")

    # 日付列のインデックス位置を取得
    date_col_index = result.columns.get_loc(date_column)

    # 年・月・日の値を計算
    year_values = result[date_column].dt.year
    month_values = result[date_column].dt.month
    day_values = result[date_column].dt.day

    # 日付列の真横に年・月・日の列を挿入
    result.insert(date_col_index + 1, f"{date_column}_year", year_values)
    result.insert(date_col_index + 2, f"{date_column}_month", month_values)
    result.insert(date_col_index + 3, f"{date_column}_day", day_values)

    # 元の日付列を削除するかどうかをユーザーに選択させる
    if remove_original is None:
        remove_choice = _get_user_choice(
            f"元の日付列 '{date_column}' を削除しますか?",
            ["削除する", "残す"]
        )
        remove_original = remove_choice == 1

    # 元の日付列を削除する場合
    if remove_original:
        result = result.drop(columns=[date_column])
        print(f"元の日付列 '{date_column}' を削除しました")
    else:
        print(f"元の日付列 '{date_column}' を保持しました")

    print(f"年・月・日列を追加しました。新しい形状: {result.shape}")

    # 結果プレビュー
    preview_cols = [c for c in result.columns if c.startswith(date_column)]
    print("\n結果プレビュー:")
    print(result[preview_cols].head())

    return result


def add_daily_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    # 日付列の選択
    date_col_choice = _get_user_choice(
        "集計の基準となる日付列を選択してください:",
        list(df.columns)
    )
    date_col = df.columns[date_col_choice - 1]

    # 集計対象列の選択
    target_cols = _get_column_choice(
        df, "集計する数値列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )

    if not target_cols:
        print("集計対象列が選択されていません。処理を中止します。")
        return df

    # 元の日付列の値を保持(文字列型)
    original_dates = df[date_col].copy()

    try:
        # 日付型に変換(一時的な処理のため)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # 無効な日付のチェック
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates}件の無効な日付を検出しました")
            print("無効な日付の行は集計から除外されます")
    except Exception as e:
        print(f"日付変換エラー: {e}")
        choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if choice == 2:
            # 元の値を復元
            df[date_col] = original_dates
            return df

    # 集計処理
    try:
        # 日付単位でグループ化し合計を計算
        daily_sums = df.groupby(date_col)[target_cols].sum()

        # 新しい列名を生成
        new_col_names = [f"{col}_daily_sum" for col in target_cols]
        daily_sums.columns = new_col_names
        daily_sums = daily_sums.reset_index()

        # 元のデータフレームにマージ
        result = pd.merge(df, daily_sums, on=date_col, how="left")

        # 元の日付列の値を復元
        result[date_col] = original_dates

        print("\n★ 日付別合計列を追加:")
        for col in new_col_names:
            print(f"  - {col}")

        # 結果プレビュー
        preview_cols = [date_col, *new_col_names]
        print("\n結果プレビュー (先頭5件):")
        print(result[preview_cols].head())

        return result

    except Exception as e:
        print(f"集計中にエラーが発生しました: {e}")
        # エラーが発生した場合も元の値を復元
        df[date_col] = original_dates
        return df


def add_monthly_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    月ごとに特定の列の合計値を計算し、元のデータフレームに新しい列として追加します。
    同じ月の各行には同じ合計値が入ります。
    """
    # 日付列の選択
    date_col_choice = _get_user_choice(
        "集計の基準となる日付列を選択してください:",
        list(df.columns)
    )
    date_col = df.columns[date_col_choice - 1]

    # 集計対象列の選択
    target_cols = _get_column_choice(
        df, "集計する数値列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )

    if not target_cols:
        print("集計対象列が選択されていません。処理を中止します。")
        return df

    # 元の日付列の値を保持(文字列型)
    original_dates = df[date_col].copy()

    try:
        # 日付型に変換
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # 年月列を追加 (YYYY-MM形式)
        df["year_month"] = df[date_col].dt.strftime("%Y-%m")

        # 無効な日付のチェック
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates}件の無効な日付を検出しました")
            print("無効な日付の行は集計から除外されます")
    except Exception as e:
        print(f"日付変換エラー: {e}")
        choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if choice == 2:
            # 元の値を復元
            df[date_col] = original_dates
            return df

    # 集計処理
    try:
        # 年月単位でグループ化し合計を計算
        monthly_sums = df.groupby("year_month")[target_cols].sum()

        # 新しい列名を生成
        new_col_names = [f"{col}_monthly_sum" for col in target_cols]
        monthly_sums.columns = new_col_names
        monthly_sums = monthly_sums.reset_index()

        # 元のデータフレームにマージ
        result = pd.merge(df, monthly_sums, on="year_month", how="left")

        # 元の日付列の値を復元
        result[date_col] = original_dates

        # 不要なyear_month列を削除
        result = result.drop(columns=["year_month"])

        print("\n★ 月別合計列を追加:")
        for col in new_col_names:
            print(f"  - {col}")

        # 結果プレビュー
        preview_cols = [date_col, *new_col_names]
        print("\n結果プレビュー (先頭5件):")
        print(result[preview_cols].head())

        return result

    except Exception as e:
        print(f"集計中にエラーが発生しました: {e}")
        # エラーが発生した場合も元の値を復元
        df[date_col] = original_dates
        return df


def plot_counter_all(
    df: pd.DataFrame,
    figsize_per_plot=(6, 4),
    max_unique_values=20,
    ncols=3
):
    """
    データフレームの全ての列についてcountplotまたはヒストグラムを表示する関数
    """
    columns = df.columns.tolist()
    n_cols = len(columns)
    if n_cols == 0:
        print("データフレームに列がありません。")
        return
    nrows = math.ceil(n_cols / ncols)
    total_figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=total_figsize)
    # Flatten axes array
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, column in enumerate(columns):
        ax = axes[i]
        unique_count = df[column].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        try:
            if unique_count > max_unique_values and is_numeric:
                ax.hist(
                    df[column].dropna(),
                    bins=30,
                    edgecolor="black"
                )
                ax.set_title(f"{column}(Histogram - {unique_count} unique)")
            elif unique_count > max_unique_values:
                top_values = df[column].value_counts().head(max_unique_values)
                top_values.plot(kind="bar", ax=ax)
                ax.set_title(f"{column}(Top {len(top_values)}/{unique_count} unique)")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
            else:
                sns.countplot(data=df, x=column, ax=ax)
                ax.set_title(f"{column}({unique_count} unique)")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
        except Exception as e:
            ax.text(
                0.5, 0.5, f"Error:{e}", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{column} (Error)")
    # Hide unused subplots
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def basic_statistics(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    数値列とカテゴリ列の基本統計量と頻度ランキングを表示する関数。
    - 数値列: count, mean, median, std, var, min, max, range, q1, q3,
      iqr, skewness, kurtosis
    - カテゴリ列: ユニーク数、上位top_nの頻度ランキング
    """
    # 数値列統計
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        print("=== 数値列の統計量 ===")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            stats = {
                "count": series.count(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "var": series.var(),
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            }
            print(f"--- {col} ---")
            for k, v in stats.items():
                if isinstance(v, int | float):
                    out = f"{k:>10}: {v:.4f}"
                else:
                    out = f"{k:>10}: {v}"
                print(out)
    else:
        print("数値列がありません。")

    # カテゴリ列頻度
    categorical_cols = [
        col
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if categorical_cols:
        print("=== カテゴリ列の頻度ランキング ===")
        for col in categorical_cols:
            counts = df[col].value_counts(dropna=False)
            print(f"--- {col} --- ユニーク数: {counts.shape[0]}")
            for i, (val, cnt) in enumerate(counts.head(top_n).items(), 1):
                pct = cnt / len(df) * 100
                print(f"{i:>2}. {val} - {cnt} ({pct:.2f}%)")
            if counts.shape[0] > top_n:
                print(f"... その他 {counts.shape[0] - top_n} 件")
    else:
        print("カテゴリ列がありません。")


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    """散布図を描画"""
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter plot of {x} vs {y}")
    plt.show()


def plot_3d_histogram(df: pd.DataFrame, x: str, y: str, bins: int = 10):
    """2変数の3次元ヒストグラムを描画"""
    hist, xedges, yedges = np.histogram2d(df[x], df[y], bins=bins)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = dy = xedges[1] - xedges[0]
    dz = hist.ravel()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    plt.show()


def save_to_database(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """
    データフレームをSQLiteデータベースに保存する関数
    """
    try:
        # 保存先テーブル名の入力
        table_name = input("\n保存先のテーブル名を入力してください: ").strip()
        if not table_name:
            print("テーブル名が入力されていません。保存を中止します。")
            return

        # 既存テーブルの確認
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if cursor.fetchone():
            overwrite = _get_user_choice(
                f"テーブル '{table_name}' は既に存在します。",
                ["上書きする", "中止する"]
            )
            if overwrite == 2:
                print("保存を中止します。")
                return
            # 既存テーブルを削除
            cursor.execute(f"DROP TABLE `{table_name}`")

        # データフレームをSQLiteに保存
        df.to_sql(table_name, conn, index=False)
        print(f"✓ データをテーブル '{table_name}' に保存しました。")
        print(f"  行数: {df.shape[0]}, 列数: {df.shape[1]}")

    except Exception as e:
        print(f"保存中にエラーが発生しました: {e}")
        conn.rollback()
        raise


def _get_type_conversion_function(type_choice: int) -> callable:
    """型変換関数を取得する"""
    if type_choice == 1:  # int
        return lambda x: pd.to_numeric(x, errors="coerce").astype("Int64")
    elif type_choice == 2:  # float
        return lambda x: pd.to_numeric(x, errors="coerce")
    elif type_choice == 3:  # str
        return lambda x: x.astype(str)
    elif type_choice == 4:  # datetime
        # 日付フォーマットの入力
        fmt = input(
            "日付フォーマットを入力 (例: %Y-%m-%d、未入力で自動判定): "
        ).strip() or None
        return lambda x, fmt=fmt: pd.to_datetime(x, format=fmt, errors="coerce")
    elif type_choice == 5:  # category
        return lambda x: x.astype("category")
    else:
        raise ValueError(f"無効な型選択: {type_choice}")


def _apply_type_conversion(
    df: pd.DataFrame,
    col: str,
    converter: callable
) -> tuple[pd.DataFrame, int]:
    """型変換を適用し、エラー数を返す"""
    result = df.copy()
    original_nulls = result[col].isna().sum()

    try:
        result[col] = converter(result[col])
        new_nulls = result[col].isna().sum()
        error_count = new_nulls - original_nulls

        print(f"✓ {col}: {df[col].dtype} → {result[col].dtype}")
        if error_count > 0:
            print(f"  警告: {error_count}件の変換エラー (NaNに変換)")

        return result, error_count
    except Exception as e:
        print(f"❌ {col} の変換に失敗: {e!s}")
        return df, len(df)


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # 変換対象列の選択 (複数選択可)
    target_cols = _get_column_choice(
        result, "型変換する列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )

    if not target_cols:
        print("変換対象列が選択されていません")
        return df

    # 変換マッピングの定義
    type_options = [
        "整数型 (int)",
        "浮動小数点数型 (float)",
        "文字列型 (str)",
        "日付型 (datetime)",
        "カテゴリ型 (category)",
    ]

    # 列ごとの変換タイプを記録
    conversion_map = {}
    error_counts = {}

    for col in target_cols:
        print(f"\n=== 列: {col} ===")
        print(f"  現在の型: {result[col].dtype}")
        print(f"  ユニーク値数: {result[col].nunique()}")
        print(f"  欠損値数: {result[col].isna().sum()}")

        # 変換タイプの選択
        type_choice = _get_user_choice("変換先のデータ型を選択:", type_options)

        # 変換関数の取得
        try:
            conversion_map[col] = _get_type_conversion_function(type_choice)
        except ValueError as e:
            print(f"❌ {e}")
            continue

    # 変換実行
    for col, converter in conversion_map.items():
        result, error_count = _apply_type_conversion(result, col, converter)
        error_counts[col] = error_count

    # エラーサマリー表示
    if any(count > 0 for count in error_counts.values()):
        print("\n=== 型変換エラーサマリー ===")
        for col, count in error_counts.items():
            if count > 0:
                print(f"  {col}: {count}件の変換エラー")

    return result


# ---main sub functions---


def _get_available_tables(conn: sqlite3.Connection) -> list[str]:
    """利用可能なテーブル一覧を取得"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def _display_tables(tables: list[str], conn: sqlite3.Connection) -> None:
    """テーブル一覧を表示"""
    print("\n利用可能なテーブル一覧:")
    for i, table in enumerate(tables, 1):
        # テーブル名をバッククォートで囲んでSQLインジェクション対策と特殊文字対応
        count = pd.read_sql_query(
            f"SELECT COUNT(*) as count FROM `{table}`", conn
        )["count"][0]
        print(f"{i}. {table} (行数: {count})")


def _execute_function(func, df: pd.DataFrame) -> pd.DataFrame:
    """選択された関数を実行"""
    if func in [plot_counter_all, basic_statistics]:
        func(df)
        return df

    if func == plot_scatter:
        x = df.columns[
            _get_user_choice("X軸の列を選択してください:", list(df.columns)) - 1
        ]
        y = df.columns[
            _get_user_choice("Y軸の列を選択してください:", list(df.columns)) - 1
        ]
        func(df, x, y)
        return df

    if func == plot_3d_histogram:
        x = df.columns[
            _get_user_choice("X軸の列を選択してください:", list(df.columns)) - 1
        ]
        y = df.columns[
            _get_user_choice("Y軸の列を選択してください:", list(df.columns)) - 1
        ]
        bins = int(input("ビン数(デフォルト10): ") or 10)
        func(df, x, y, bins)
        return df

    result = func(df)
    return result if isinstance(result, pd.DataFrame) else df


# ---main sub functions ---


def run_pretreatment():
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
        print(
            f"\nテーブル '{selected_table}' を読み込みました "
            f"(行: {df.shape[0]}, 列: {df.shape[1]})"
        )

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
            choice = _get_user_choice(
                "適用する前処理関数を選択してください:", func_choices
            )

            if choice == len(func_choices) - 1:  # 保存して終了
                save_to_database(df, conn)
                break
            if choice == len(func_choices):  # 保存せずに終了
                if _get_user_choice(
                    "変更を保存せずに終了しますか?", ["はい", "いいえ"]
                ) == 1:
                    break
                continue

            func = available_funcs[choice - 1]
            print(f"\n選択した関数: {func.__name__}")

            try:
                df = _execute_function(func, df)
                print(f"\n処理完了: {func.__name__}")
                print(f"現在のデータ形状: {df.shape}")

                if _get_user_choice(
                    "この時点でデータを保存しますか?", ["いいえ", "はい"]
                ) == 2:
                    save_to_database(df, conn)

            except Exception as e:
                print(f"関数実行中にエラーが発生しました: {e}")

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("\nデータベース接続を閉じました")
        print("プログラムを終了します")


if __name__ == "__main__":
    run_pretreatment()
