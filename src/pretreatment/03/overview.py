import pandas as pd


def overview_data(df: pd.DataFrame) -> None:
    def _get_column_types(df: pd.DataFrame):
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        missing_cols = df.columns[df.isna().any()].tolist()
        return {"numeric": numeric_cols, "categorical": categorical_cols, "missing": missing_cols}

    col_types = _get_column_types(df)

    print("=== データ概要 ===")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"数値列: {len(col_types['numeric'])}")
    print(f"カテゴリ列: {len(col_types['categorical'])}")
    print(f"欠損値あり列: {len(col_types['missing'])}")

    cols_input = input("表示する列をカンマ区切りで入力 (例: col1,col2)\n全て表示: all, 終了: exit\n入力: ").strip()

    if cols_input.lower() == "exit":
        return
    if cols_input.lower() == "all":
        print("\n全列を表示:")
        print(df.head())
        return

    selected_cols = [col.strip() for col in cols_input.split(",") if col.strip()]
    valid_cols = [col for col in selected_cols if col in df.columns]
    invalid_cols = set(selected_cols) - set(valid_cols)

    if invalid_cols:
        print(f"次の列は存在しません: {', '.join(invalid_cols)}")
    if not valid_cols:
        print("有効な列が選択されていません")
        return

    print(f"\n選択列 ({', '.join(valid_cols)}) の先頭5行:")
    print(df[valid_cols].head())
    cont = input("\n他の列を表示しますか? (y/n): ").strip().lower()
    if cont != "y":
        return
