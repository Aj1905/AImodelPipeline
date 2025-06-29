import pandas as pd


def _handle_condition_deletion(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """条件に基づく行削除を実行"""
    try:
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


def _handle_duplicate_deletion(df: pd.DataFrame, subset: list[str] | None) -> pd.DataFrame:
    """重複行の削除を実行"""
    original_count = len(df)
    df = df.drop_duplicates(subset=subset) if subset else df.drop_duplicates()
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
