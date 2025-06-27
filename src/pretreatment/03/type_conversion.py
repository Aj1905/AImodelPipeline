import pandas as pd

from .utils import _get_column_choice, _get_user_choice


def _get_type_conversion_function(type_choice: int) -> callable:
    if type_choice == 1:
        return lambda x: pd.to_numeric(x, errors="coerce").astype("Int64")
    elif type_choice == 2:
        return lambda x: pd.to_numeric(x, errors="coerce")
    elif type_choice == 3:
        return lambda x: x.astype(str)
    elif type_choice == 4:
        fmt = input("日付フォーマットを入力 (例: %Y-%m-%d、未入力で自動判定): ").strip() or None
        return lambda x, fmt=fmt: pd.to_datetime(x, format=fmt, errors="coerce")
    elif type_choice == 5:
        return lambda x: x.astype("category")
    else:
        raise ValueError(f"無効な型選択: {type_choice}")


def _apply_type_conversion(df: pd.DataFrame, col: str, converter: callable) -> tuple[pd.DataFrame, int]:
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
    target_cols = _get_column_choice(
        result, "型変換する列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )
    if not target_cols:
        print("変換対象列が選択されていません")
        return df
    type_options = [
        "整数型 (int)",
        "浮動小数点数型 (float)",
        "文字列型 (str)",
        "日付型 (datetime)",
        "カテゴリ型 (category)",
    ]
    conversion_map: dict[str, callable] = {}
    error_counts: dict[str, int] = {}
    for col in target_cols:
        print(f"\n=== 列: {col} ===")
        print(f"  現在の型: {result[col].dtype}")
        print(f"  ユニーク値数: {result[col].nunique()}")
        print(f"  欠損値数: {result[col].isna().sum()}")
        type_choice = _get_user_choice("変換先のデータ型を選択:", type_options)
        try:
            conversion_map[col] = _get_type_conversion_function(type_choice)
        except ValueError as e:
            print(f"❌ {e}")
            continue
    for col, converter in conversion_map.items():
        result, error_count = _apply_type_conversion(result, col, converter)
        error_counts[col] = error_count
    if any(count > 0 for count in error_counts.values()):
        print("\n=== 型変換エラーサマリー ===")
        for col, count in error_counts.items():
            if count > 0:
                print(f"  {col}: {count}件の変換エラー")
    return result
