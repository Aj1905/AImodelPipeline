import pandas as pd


def delete_columns(df: pd.DataFrame, columns: list | str | None = None) -> pd.DataFrame:
    """DataFrameから指定した列を削除する"""
    if columns is None:
        print("\n📋 現在の列一覧:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print("\n削除方法を選択:")
        print("  1. 削除する列を指定")
        print("  2. 残す列を指定")
        while True:
            mode_input = input("選択 (1 or 2): ").strip()
            if mode_input in ["1", "2"]:
                is_delete_mode = mode_input == "1"
                break
            print("❌ 1 または 2 を入力してください")
        action = "削除" if is_delete_mode else "残す"
        while True:
            user_input = input(f"\n{action}列番号を入力 (例: 1,3,5): ").strip()
            if not user_input:
                print("処理を中止します。")
                return df
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(",")]
                if any(i < 0 or i >= len(df.columns) for i in indices):
                    raise IndexError
                selected_cols = [df.columns[i] for i in indices]
                break
            except (ValueError, IndexError):
                print(f"❌ 無効な入力です。1~{len(df.columns)}の範囲で入力してください")
        cols_to_drop = selected_cols if is_delete_mode else [col for col in df.columns if col not in selected_cols]
    else:
        if isinstance(columns, str):
            columns = [columns]
        cols_to_drop = columns
    result = df.drop(columns=cols_to_drop)
    print(f"\n✅ 削除完了: {cols_to_drop}")
    print(f"📊 新しい形状: {result.shape}")
    return result
