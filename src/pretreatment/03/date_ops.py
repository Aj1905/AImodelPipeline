import jpholiday
import pandas as pd

from .utils import _get_column_choice, _get_user_choice


def convert_date_format(
    df: pd.DataFrame, column_name: str | None = None, input_format: str | None = None
) -> pd.DataFrame:
    """日付データを指定形式に変換する"""
    result = df.copy()
    if column_name is None:
        col_choice = _get_user_choice("日付データの列を選択してください:", list(df.columns))
        column_name = df.columns[col_choice - 1]
    if column_name not in result.columns:
        print(f"エラー: 列 '{column_name}' が存在しません")
        return df
    input_format = input("日付フォーマットを入力 (例: %Y-%m-%d、未入力で自動判定): ").strip() or None
    try:
        result[column_name] = pd.to_datetime(result[column_name], format="%Y-%m-%d", errors="coerce")
        print(f"✓ 日付変換完了: {column_name} 列 ({input_format} → %Y-%m-%d)")
    except Exception as e:
        print(f"✗ 変換エラー: {e}")
        print("入力形式パターンを確認してください")
        return df
    return result


def add_weekday_column(
    df: pd.DataFrame, date_column: str | None = None, weekday_column_name: str = "曜日"
) -> pd.DataFrame:
    result = df.copy()
    if date_column is None:
        print(f"\nデータ行数: {len(result)}, 列数: {len(result.columns)}")
        col_choice = _get_user_choice("日付列を選択してください(%Y-%m-%d形式):", list(result.columns))
        date_column = result.columns[col_choice - 1]
    if date_column not in result.columns:
        print(f"エラー: 列 '{date_column}' が存在しません")
        return df
    if weekday_column_name is None or weekday_column_name == "曜日":
        name = input("追加する曜日列の名前を入力してください (デフォルト: '曜日'): ").strip()
        if name:
            weekday_column_name = name
    try:
        pd.to_datetime(result[date_column].dropna())
    except Exception as e:
        print(f"⚠ 警告: '{date_column}' 列の値が日付として認識できません: {e}")
        continue_choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if continue_choice == 2:
            return df
    result[weekday_column_name] = pd.to_datetime(result[date_column]).dt.day_name()
    print(f"✓ 曜日列 '{weekday_column_name}' を追加しました。")
    preview = result[[date_column, weekday_column_name]].head()
    print("結果プレビュー:")
    print(preview)
    return result


def add_holiday_flag_column(
    df: pd.DataFrame, date_column: str | None = None, holiday_column_name: str = "祝日フラグ"
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
        name = input("追加する祝日フラグ列の名前を入力してください (デフォルト: '祝日フラグ'): ").strip()
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
    result[holiday_column_name] = dates.apply(lambda d: bool(d and jpholiday.is_holiday(d)))
    print(f"✓ 祝日フラグ列 '{holiday_column_name}' を追加しました。")
    preview = result[[date_column, holiday_column_name]].head()
    print("結果プレビュー:")
    print(preview)
    return result


def add_year_month_day(
    df: pd.DataFrame, date_column: str | None = None, remove_original: bool | None = None
) -> pd.DataFrame:
    result = df.copy()
    if date_column is None:
        print(f"\nデータ行数: {len(result)}, 列数: {len(result.columns)}")
        col_choice = _get_user_choice("日付列を選択してください(%Y-%m-%d形式):", list(result.columns))
        date_column = result.columns[col_choice - 1]
    if date_column not in result.columns:
        print(f"エラー: 列 '{date_column}' が存在しません")
        return df
    try:
        result[date_column] = pd.to_datetime(result[date_column], format="%Y-%m-%d", errors="coerce")
    except Exception as e:
        print(f"日付変換エラー: {e}")
        return df
    invalid_count = result[date_column].isna().sum()
    if invalid_count > 0:
        print(f"警告: {invalid_count}件の無効な日付形式をNaNに変換しました")
    date_col_index = result.columns.get_loc(date_column)
    result.insert(date_col_index + 1, f"{date_column}_year", result[date_column].dt.year)
    result.insert(date_col_index + 2, f"{date_column}_month", result[date_column].dt.month)
    result.insert(date_col_index + 3, f"{date_column}_day", result[date_column].dt.day)
    if remove_original is None:
        remove_choice = _get_user_choice(f"元の日付列 '{date_column}' を削除しますか?", ["削除する", "残す"])
        remove_original = remove_choice == 1
    if remove_original:
        result = result.drop(columns=[date_column])
        print(f"元の日付列 '{date_column}' を削除しました")
    else:
        print(f"元の日付列 '{date_column}' を保持しました")
    print(f"年・月・日列を追加しました。新しい形状: {result.shape}")
    preview_cols = [c for c in result.columns if c.startswith(date_column)]
    print("\n結果プレビュー:")
    print(result[preview_cols].head())
    return result


def add_daily_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    date_col_choice = _get_user_choice("集計の基準となる日付列を選択してください:", list(df.columns))
    date_col = df.columns[date_col_choice - 1]
    target_cols = _get_column_choice(
        df, "集計する数値列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )
    if not target_cols:
        print("集計対象列が選択されていません。処理を中止します。")
        return df
    original_dates = df[date_col].copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates}件の無効な日付を検出しました")
            print("無効な日付の行は集計から除外されます")
    except Exception as e:
        print(f"日付変換エラー: {e}")
        choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if choice == 2:
            df[date_col] = original_dates
            return df
    try:
        daily_sums = df.groupby(date_col)[target_cols].sum()
        new_col_names = [f"{col}_daily_sum" for col in target_cols]
        daily_sums.columns = new_col_names
        daily_sums = daily_sums.reset_index()
        result = pd.merge(df, daily_sums, on=date_col, how="left")
        result[date_col] = original_dates
        print("\n★ 日付別合計列を追加:")
        for col in new_col_names:
            print(f"  - {col}")
        preview_cols = [date_col, *new_col_names]
        print("\n結果プレビュー (先頭5件):")
        print(result[preview_cols].head())
        return result
    except Exception as e:
        print(f"集計中にエラーが発生しました: {e}")
        df[date_col] = original_dates
        return df


def add_monthly_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    date_col_choice = _get_user_choice("集計の基準となる日付列を選択してください:", list(df.columns))
    date_col = df.columns[date_col_choice - 1]
    target_cols = _get_column_choice(
        df, "集計する数値列を選択してください (複数選択可):", allow_multiple=True, allow_all=False
    )
    if not target_cols:
        print("集計対象列が選択されていません。処理を中止します。")
        return df
    original_dates = df[date_col].copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year_month"] = df[date_col].dt.strftime("%Y-%m")
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates}件の無効な日付を検出しました")
            print("無効な日付の行は集計から除外されます")
    except Exception as e:
        print(f"日付変換エラー: {e}")
        choice = _get_user_choice("処理を続行しますか?", ["続行する", "中止する"])
        if choice == 2:
            df[date_col] = original_dates
            return df
    try:
        monthly_sums = df.groupby("year_month")[target_cols].sum()
        new_col_names = [f"{col}_monthly_sum" for col in target_cols]
        monthly_sums.columns = new_col_names
        monthly_sums = monthly_sums.reset_index()
        result = pd.merge(df, monthly_sums, on="year_month", how="left")
        result[date_col] = original_dates
        result = result.drop(columns=["year_month"])
        print("\n★ 月別合計列を追加:")
        for col in new_col_names:
            print(f"  - {col}")
        preview_cols = [date_col, *new_col_names]
        print("\n結果プレビュー (先頭5件):")
        print(result[preview_cols].head())
        return result
    except Exception as e:
        print(f"集計中にエラーが発生しました: {e}")
        df[date_col] = original_dates
        return df
