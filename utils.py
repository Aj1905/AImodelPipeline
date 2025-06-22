import sys

import numpy as np
import polars as pl
from sklearn.metrics import r2_score


def _get_user_choice(prompt: str, options: list[str]) -> int:
    """ユーザーに選択肢を提示し、選択された番号を返す"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = int(input("選択してください: ").strip())
            if 1 <= choice <= len(options):
                return choice
            else:
                print(f"1~{len(options)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _select_target_and_features(data: pl.DataFrame) -> tuple[str, list[str]]:
    """目的変数と特徴量を選択"""
    print("\n利用可能な目的変数候補:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i}. {col}")

    while True:
        try:
            target_sel = int(input("\n目的変数の番号を入力してください: ").strip())
            if 1 <= target_sel <= len(data.columns):
                target_column = data.columns[target_sel - 1]
                break
            else:
                print(f"1~{len(data.columns)}の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")

    print(f"選択された目的変数: {target_column}")

    available = [c for c in data.columns if c != target_column]
    feature_columns = select_features_interactively(data.select(available))
    print(f"選択特徴量: {feature_columns}\n")

    return target_column, feature_columns


def select_features_interactively(df: pl.DataFrame) -> list:
    """データフレームの列一覧を番号選択"""
    cols = df.columns
    print("\n利用可能な特徴量候補:")
    for i, c in enumerate(cols, 1):
        print(f"  {i}. {c}")
    sel = input("使用する特徴量番号をカンマ区切りで入力してください: ")
    try:
        idxs = [int(x.strip()) - 1 for x in sel.split(",")]
        chosen = [cols[i] for i in idxs if 0 <= i < len(cols)]
        if not chosen:
            raise ValueError
        return chosen
    except Exception:
        print("無効な選択です。")
        sys.exit(1)


def validate_columns(df: pl.DataFrame, required_columns: list[str]) -> bool:
    """列の存在チェック"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"エラー: 以下の列が存在しません: {missing}")
        print(f"利用可能な列: {list(df.columns)}")
        return False
    return True


def check_division_by_zero(operation: str, col2_num: pl.Series) -> None:
    """ゼロ除算の警告"""
    if operation == "/" and (col2_num == 0).any():
        zero_count = (col2_num == 0).sum()
        print(f"警告: ゼロ除算が発生する可能性があります({zero_count} 個のゼロ)")


def execute_horizontal_concatenate(
    merged_df: pl.DataFrame, current_df: pl.DataFrame, prev_table: str, current_table: str
) -> pl.DataFrame:
    """HORIZONTAL_CONCATENATEを実行"""
    print(f"HORIZONTAL_CONCATENATE実行: {prev_table} | {current_table}")
    if len(merged_df) == len(current_df):
        right_cols = current_df.columns
        rename_dict = {col: f"{current_table}_{col}" for col in right_cols}
        renamed_df = current_df.rename(rename_dict)
        result = pl.concat([merged_df, renamed_df], how="horizontal")
        print(f"→ HORIZONTAL_CONCATENATE後: {result.shape}")
        return result
    else:
        print("❌ 行数が異なるためHORIZONTAL_CONCATENATEできません")
        print(f"  {prev_table}の行数: {len(merged_df)}")
        print(f"  {current_table}の行数: {len(current_df)}")
        return merged_df


def find_common_columns(dfs: dict[str, pl.DataFrame]) -> dict:
    """テーブル間の共通カラムを発見"""
    common_columns = {}
    table_names = list(dfs.keys())

    for i in range(len(table_names) - 1):
        table1 = table_names[i]
        table2 = table_names[i + 1]

        cols1 = set(dfs[table1].columns)
        cols2 = set(dfs[table2].columns)
        common = cols1 & cols2

        if not common:
            result = _handle_no_common_columns(dfs, table1, table2)
            common_columns[(table1, table2)] = result
        else:
            print(f"\n{table1} と {table2} の共通カラム:")
            for j, col in enumerate(common, 1):
                print(f"  {j}. {col}")

            while True:
                try:
                    choice = int(input(f"結合キーとして使用するカラム番号を入力 (1-{len(common)}): ").strip())
                    if 1 <= choice <= len(common):
                        join_key = list(common)[choice - 1]
                        common_columns[(table1, table2)] = join_key
                        break
                    else:
                        print(f"1~{len(common)}の範囲で入力してください")
                except ValueError:
                    print("数値を入力してください")

    return common_columns


def _handle_no_common_columns(dfs: dict[str, pl.DataFrame], table1: str, table2: str) -> str:
    """共通カラムがない場合の処理"""
    print(f"\n{table1} と {table2} に共通カラムがありません")
    print("結合方法を選択してください:")
    print("1. スキップ")
    print("2. クロス結合")
    print("3. 縦方向結合")
    print("4. 横方向結合")
    print("5. 手動指定")

    while True:
        try:
            choice = int(input("選択 (1-5): ").strip())
            if choice == 1:
                return "SKIP"
            elif choice == 2:
                return "CROSS_JOIN"
            elif choice == 3:
                return "CONCATENATE"
            elif choice == 4:
                return "HORIZONTAL_CONCATENATE"
            elif choice == 5:
                return _get_manual_join_keys(dfs, table1, table2)
            else:
                print("1~5の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_manual_join_keys(dfs: dict[str, pl.DataFrame], table1: str, table2: str) -> tuple[str, str]:
    """手動で結合キーを指定"""
    print(f"\n{table1}のカラム:")
    for i, col in enumerate(dfs[table1].columns, 1):
        print(f"  {i}. {col}")

    print(f"\n{table2}のカラム:")
    for i, col in enumerate(dfs[table2].columns, 1):
        print(f"  {i}. {col}")

    while True:
        try:
            key1_idx = int(input(f"{table1}の結合キー番号: ").strip()) - 1
            key2_idx = int(input(f"{table2}の結合キー番号: ").strip()) - 1

            if 0 <= key1_idx < len(dfs[table1].columns) and 0 <= key2_idx < len(dfs[table2].columns):
                key1 = dfs[table1].columns[key1_idx]
                key2 = dfs[table2].columns[key2_idx]
                return (key1, key2)
            else:
                print("有効な範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def analyze_train_test_comparison(
    fold_details: list[dict],
    cv_method: str,
    all_y_true: list[float],
    all_y_pred: list[float],
    scores: list[float] | None = None,
    cv_config: dict | None = None,
) -> None:
    """学習データとテストデータの比較分析"""
    print("\n=== 学習データ・テストデータ比較分析 ===")

    overall_r2 = r2_score(all_y_true, all_y_pred)
    errors = np.array(all_y_true) - np.array(all_y_pred)

    print("1. 全体的な性能指標")
    print("-" * 40)
    print(f"全体R²スコア: {overall_r2:.4f}")
    print(f"平均絶対誤差: {np.mean(np.abs(errors)):.4f}")
    print(f"二乗平均平方根誤差: {np.sqrt(np.mean(errors**2)):.4f}")

    print("\n2. 各Foldの詳細結果")
    print("-" * 40)

    for fold in fold_details:
        print(f"\nFold {fold['fold']}:")
        print(
            f"  学習データ: {fold['train_size']}行, 平均: {fold['train_stats']['mean']:.4f}, 標準偏差: {fold['train_stats']['std']:.4f}"
        )
        print(
            f"  テストデータ: {fold['test_size']}行, 平均: {fold['test_stats']['mean']:.4f}, 標準偏差: {fold['test_stats']['std']:.4f}"
        )
        print(f"  R²スコア: {fold['score']:.4f}")

        mean_diff = abs(fold["train_stats"]["mean"] - fold["test_stats"]["mean"])
        std_diff = abs(fold["train_stats"]["std"] - fold["test_stats"]["std"])

        if mean_diff < 0.1 and std_diff < 0.1:
            print("  ✓ 学習・テストデータの分布は類似")
        elif mean_diff > 0.5 or std_diff > 0.5:
            print("  ⚠️ 学習・テストデータの分布に大きな差")
        else:
            print("  ○ 学習・テストデータの分布に中程度の差")

    print("\n3. 推奨事項")
    print("-" * 40)

    if overall_r2 > 0.8:
        print("• 全体的な予測性能は良好です")
    elif overall_r2 > 0.6:
        print("• 予測性能は中程度です。特徴量エンジニアリングの改善を検討してください")
    else:
        print("• 予測性能が低いです。モデルと特徴量の見直しを推奨します")

    error_std = np.std(errors)
    if error_std < 0.1:
        print("• 予測誤差は安定しています")
    else:
        print("• 予測誤差の変動が大きいです。モデルの安定性向上を検討してください")
