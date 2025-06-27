import pandas as pd

from .utils import _get_user_choice


def replace_text(
    df: pd.DataFrame,
    column: str | None = None,
    to_replace: str | None = None,
    value: str | None = None,
) -> pd.DataFrame:
    if to_replace is None:
        to_replace = input("検索文字列を入力してください: ").strip()
    if column is None:
        cols_with_all = [*list(df.columns), "すべての列"]
        col_choice = _get_user_choice("置換対象の列を選択してください:", cols_with_all)
        column = df.columns[col_choice - 1] if col_choice <= len(df.columns) else None
    if value is None:
        value = input("置換後の文字列を入力してください: ").strip()

    result = df.copy()
    if column:
        if column not in result.columns:
            print(f"警告: 列 '{column}' が存在しません。")
            print(f"利用可能列: {list(result.columns)}")
            return result
        target_cols = [column]
    else:
        target_cols = result.columns.tolist()

    total_replaced = 0
    for col in target_cols:
        before = result[col].astype(str)
        after = before.str.replace(to_replace, value, regex=False)
        count = (before != after).sum()
        total_replaced += count
        result[col] = after

    print(f"置換完了: '{to_replace}' → '{value}' | 合計置換数: {total_replaced}")
    return result
