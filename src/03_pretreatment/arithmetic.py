import pandas as pd

from .utils import _get_user_choice, check_division_by_zero, validate_columns


def _get_column_names(df: pd.DataFrame, col1: str | None, col2: str | None) -> tuple[str, str]:
    if col1 is None:
        col1_choice = _get_user_choice("計算に使用する列1を選択してください:", list(df.columns))
        col1 = df.columns[col1_choice - 1]
    if col2 is None:
        col2_choice = _get_user_choice("計算に使用する列2を選択してください:", list(df.columns))
        col2 = df.columns[col2_choice - 1]
    return col1, col2


def _validate_columns(df: pd.DataFrame, col1: str, col2: str) -> bool:
    return validate_columns(df, [col1, col2])


def _get_operation(operation: str | None) -> str:
    if operation is None:
        operations = ["+", "-", "*", "/"]
        op_choice = _get_user_choice("演算を選択してください:", operations)
        operation = operations[op_choice - 1]
    return operation


def _get_new_column_name(df: pd.DataFrame, new_col: str | None = None) -> str:
    if new_col is None:
        new_col = input("生成する新列名を入力してください: ").strip()
        if not new_col:
            raise ValueError("新列名が入力されていません。")
    if new_col in df.columns:
        overwrite_choice = _get_user_choice(
            f"列 '{new_col}' は既に存在します。",
            ["上書きする", "処理を中止"],
        )
        if overwrite_choice == 2:
            raise ValueError("処理を中止します。")
    return new_col


def _prepare_numeric_data(df: pd.DataFrame, col1: str, col2: str) -> tuple[pd.Series, pd.Series]:
    col1_num = pd.to_numeric(df[col1], errors="coerce")
    col2_num = pd.to_numeric(df[col2], errors="coerce")
    for name, col_data in [(col1, col1_num), (col2, col2_num)]:
        nans = col_data.isna().sum()
        if nans > 0:
            print(f"警告: 列 '{name}' に数値変換できない値が {nans} 個あります")
    return col1_num, col2_num


def _check_division_by_zero(operation: str, col2_num: pd.Series) -> None:
    check_division_by_zero(operation, col2_num)


def _perform_calculation(col1_num: pd.Series, col2_num: pd.Series, operation: str) -> pd.Series:
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return operations[operation](col1_num, col2_num)


def _show_statistics(result: pd.DataFrame, col1: str, col2: str, operation: str, new_col: str) -> None:
    stats = result[new_col].describe()
    print(f"★ {col1} {operation} {col2} を '{new_col}' 列に追加しました。")
    print(f"  count={stats['count']:.0f}, mean={stats['mean']:.2f}, ")
    print(f"  min={stats['min']:.2f}, max={stats['max']:.2f}")
    nan_cnt = result[new_col].isna().sum()
    if nan_cnt > 0:
        print(f"  NaN (計算不能) が {nan_cnt} 個あります")


def arithmetic_columns(
    df: pd.DataFrame,
    col1: str | None = None,
    col2: str | None = None,
    operation: str | None = None,
    new_col: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    col1, col2 = _get_column_names(df, col1, col2)
    if not _validate_columns(result, col1, col2):
        return df
    operation = _get_operation(operation)
    new_col = _get_new_column_name(result, new_col)
    if new_col is None:
        return df
    col1_num, col2_num = _prepare_numeric_data(result, col1, col2)
    _check_division_by_zero(operation, col2_num)
    result[new_col] = _perform_calculation(col1_num, col2_num, operation)
    _show_statistics(result, col1, col2, operation, new_col)
    return result
