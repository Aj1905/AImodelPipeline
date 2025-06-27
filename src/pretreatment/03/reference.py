import pandas as pd

from .arithmetic import _get_new_column_name
from .utils import _get_user_choice


def _get_mapping_from_user(
    search_list: list[str] | None = None, replace_list: list[str] | None = None
) -> tuple[list[str], list[str]]:
    if search_list is None:
        search_input = input("変換前の文字列をカンマ区切りで入力してください: ").strip()
        if not search_input:
            raise ValueError("変換前リストが入力されていません。")
        search_list = [s.strip() for s in search_input.split(",")]
    if replace_list is None:
        replace_input = input("変換後の文字列をカンマ区切りで入力してください: ").strip()
        if not replace_input:
            raise ValueError("変換後リストが入力されていません。")
        replace_list = [s.strip() for s in replace_input.split(",")]
    if len(search_list) != len(replace_list):
        raise ValueError(f"変換前({len(search_list)})と変換後({len(replace_list)})の要素数が一致しません")
    return search_list, replace_list


def _apply_mapping(df: pd.DataFrame, ref_col: str, mapping: dict[str, str], new_col: str) -> pd.DataFrame:
    result = df.copy()
    result[new_col] = result[ref_col].astype(str).map(mapping)
    unmapped = [v for v in result[ref_col].astype(str).unique() if v not in mapping]
    if unmapped:
        print(f"マッピングにない値: {unmapped}")
        dv = input("未マッピング値のデフォルトを入力(空白でNaN): ").strip()
        if dv:
            result[new_col] = result[new_col].fillna(dv)
    cols = list(result.columns)
    idx = cols.index(ref_col) + 1
    cols.remove(new_col)
    cols.insert(idx, new_col)
    result = result[cols]
    return result


def generate_column_from_reference(
    df: pd.DataFrame,
    ref_col: str | None = None,
    search_list: list[str] | None = None,
    replace_list: list[str] | None = None,
    new_col: str | None = None,
) -> pd.DataFrame:
    try:
        if ref_col is None:
            ref_col_choice = _get_user_choice("参照する列を選択してください:", list(df.columns))
            ref_col = df.columns[ref_col_choice - 1]
        if ref_col not in df.columns:
            raise ValueError(f"列 '{ref_col}' が存在しません。利用可能列: {list(df.columns)}")
        search_list, replace_list = _get_mapping_from_user(search_list, replace_list)
        mapping = dict(zip(search_list, replace_list, strict=False))
        print(f"マッピング: {mapping}")
        new_col = _get_new_column_name(df, new_col)
        result = _apply_mapping(df, ref_col, mapping, new_col)
        cnt_s = result[new_col].notna().sum()
        cnt_nan = result[new_col].isna().sum()
        print(f"★ 新列 '{new_col}' を追加しました。")
        print(f"  成功: {cnt_s}, NaN: {cnt_nan}, shape: {result.shape}")
        return result
    except ValueError as e:
        print(f"エラー: {e!s}")
        return df
