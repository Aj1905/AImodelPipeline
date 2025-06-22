import math
import pickle

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from .data_loader import load_and_join_tables, select_tables_interactively
from .utils import _get_user_choice


class PreprocessingConfig:
    """前処理設定を管理するクラス"""

    def __init__(self):
        self.encoding_config = {}
        self.polynomial_config = {}
        self.arithmetic_config = {}
        self.feature_names = []

    def save(self, filepath: str):
        """設定をファイルに保存"""
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath: str):
        """設定をファイルから読み込み"""
        with open(filepath, "rb") as f:
            self.__dict__.update(pickle.load(f))


def _load_and_preprocess_data(db_path: str) -> tuple[pl.DataFrame, list[str]]:
    """データ読み込みと前処理を実行"""
    print("=== データ読み込み ===")

    selected_tables = select_tables_interactively(db_path)
    data = load_and_join_tables(db_path, selected_tables)

    print(f"データ読み込み完了: {data.shape}")

    processed_data = execute_preprocessing_loop(data)

    return processed_data, selected_tables


def _display_categorical_stats(df: pl.DataFrame, categorical_cols: list[str]) -> None:
    """カテゴリカル変数の統計情報を表示"""
    print("\n=== カテゴリカル変数の統計情報 ===")
    for col in categorical_cols:
        unique_count = df[col].n_unique()
        print(f"{col}: {unique_count}個のユニーク値")


def _apply_encoding(df: pl.DataFrame, col: str, encoding_type: str) -> pl.DataFrame:
    """指定されたエンコーディングを適用"""
    df_pd = df.to_pandas()

    if encoding_type == "label":
        le = LabelEncoder()
        df_pd[f"{col}_encoded"] = le.fit_transform(df_pd[col].astype(str))
        print(f"  ✓ {col} → {col}_encoded (Label Encoding)")

    elif encoding_type == "onehot":
        dummies = pd.get_dummies(df_pd[col], prefix=col)
        df_pd = pd.concat([df_pd, dummies], axis=1)
        print(f"  ✓ {col} → {len(dummies.columns)}個のダミー変数")

    elif encoding_type == "frequency":
        freq_map = df_pd[col].value_counts().to_dict()
        df_pd[f"{col}_freq"] = df_pd[col].map(freq_map)
        print(f"  ✓ {col} → {col}_freq (Frequency Encoding)")

    return pl.from_pandas(df_pd)


def _get_categorical_columns(df: pl.DataFrame) -> list[str]:
    """カテゴリカル変数の候補を取得"""
    categorical_cols = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in ["Utf8", "String"] or df[col].n_unique() < 50:
            categorical_cols.append(col)

    return categorical_cols


def _get_encoding_config(categorical_cols: list[str]) -> dict:
    """エンコーディング設定を取得"""
    print(f"\n{len(categorical_cols)}個のカテゴリカル変数が検出されました:")
    for i, col in enumerate(categorical_cols, 1):
        print(f"  {i}. {col}")

    encoding_options = [
        "Label Encoding - 数値ラベルに変換",
        "One-Hot Encoding - ダミー変数に変換",
        "Frequency Encoding - 出現頻度に変換",
        "スキップ - エンコーディングしない",
    ]

    encoding_config = {}

    for col in categorical_cols:
        print(f"\n'{col}' のエンコーディング方法を選択:")
        choice = _get_user_choice("エンコーディング方法:", encoding_options)

        if choice == 1:
            encoding_config[col] = "label"
        elif choice == 2:
            encoding_config[col] = "onehot"
        elif choice == 3:
            encoding_config[col] = "frequency"
        else:
            encoding_config[col] = "skip"

    return encoding_config


def unified_encoding(df: pl.DataFrame, config: PreprocessingConfig = None) -> pl.DataFrame:
    """統一されたカテゴリカル変数エンコーディング"""
    print("\n=== カテゴリカル変数エンコーディング ===")

    categorical_cols = _get_categorical_columns(df)

    if not categorical_cols:
        print("カテゴリカル変数が見つかりませんでした")
        return df

    _display_categorical_stats(df, categorical_cols)
    encoding_config = _get_encoding_config(categorical_cols)

    # 設定を記録
    if config is not None:
        config.encoding_config = encoding_config

    processed_df = df.clone()

    for col, encoding_type in encoding_config.items():
        if encoding_type != "skip":
            processed_df = _apply_encoding(processed_df, col, encoding_type)

    print(f"\n✓ エンコーディング完了: {df.shape} → {processed_df.shape}")
    return processed_df


def _display_polynomial_stats(df: pl.DataFrame, numeric_cols: list[str]) -> None:
    """多項式特徴量生成前の統計情報を表示"""
    print(f"\n数値列数: {len(numeric_cols)}")
    print("多項式特徴量生成により列数が大幅に増加する可能性があります")


def _calculate_polynomial_counts(n_features: int, degree: int) -> int:
    """多項式特徴量の数を計算"""
    return math.comb(n_features + degree, degree) - 1


def _generate_polynomial_features(
    df: pl.DataFrame, numeric_cols: list[str], degree: int, interaction_only: bool
) -> pl.DataFrame:
    """多項式特徴量を生成"""
    df_pd = df.to_pandas()

    if not numeric_cols:
        print("数値列が見つかりませんでした")
        return df

    x = df_pd[numeric_cols]

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    x_poly = poly.fit_transform(x)
    feature_names = poly.get_feature_names_out(numeric_cols)

    poly_df = pd.DataFrame(x_poly, columns=feature_names, index=df_pd.index)

    new_cols = [col for col in poly_df.columns if col not in numeric_cols]

    result_df = df_pd.copy()
    for col in new_cols:
        result_df[col] = poly_df[col]

    print(f"  ✓ 生成された新しい特徴量: {len(new_cols)}個")
    print(f"  ✓ 総特徴量数: {len(result_df.columns)}個")

    if len(new_cols) > 0:
        print(f"  新しい特徴量の例: {new_cols[:5]}")

    return pl.from_pandas(result_df)


def polynomial_features(df: pl.DataFrame, config: PreprocessingConfig = None) -> pl.DataFrame:
    """多項式特徴量生成"""
    print("\n=== 多項式特徴量生成 ===")

    numeric_cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in ["Int64", "Float64", "Int32", "Float32"]:
            numeric_cols.append(col)

    if not numeric_cols:
        print("数値列が見つかりませんでした")
        return df

    _display_polynomial_stats(df, numeric_cols)

    degree_options = [
        "2次 - 2乗項と交互作用項を生成",
        "3次 - 3乗項まで生成(計算量大)",
        "交互作用のみ - 掛け算項のみ生成",
        "スキップ - 多項式特徴量を生成しない",
    ]

    choice = _get_user_choice("多項式特徴量の設定:", degree_options)

    if choice == 1:
        degree = 2
        interaction_only = False
    elif choice == 2:
        degree = 3
        interaction_only = False
    elif choice == 3:
        degree = 2
        interaction_only = True
    else:
        print("多項式特徴量生成をスキップします")
        return df

    # 設定を記録
    if config is not None:
        config.polynomial_config = {"degree": degree, "interaction_only": interaction_only}

    expected_features = _calculate_polynomial_counts(len(numeric_cols), degree)
    if not interaction_only and expected_features > 1000:
        print(f"⚠️ 警告: 生成される特徴量数が{expected_features}個と多くなります")
        confirm = input("続行しますか? (y/n): ").strip().lower()
        if confirm != "y":
            print("多項式特徴量生成をキャンセルしました")
            return df

    result_df = _generate_polynomial_features(df, numeric_cols, degree, interaction_only)

    print(f"\n✓ 多項式特徴量生成完了: {df.shape} → {result_df.shape}")
    return result_df


def _get_column_names(df: pl.DataFrame) -> list[str]:
    """数値列の名前を取得"""
    numeric_cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in ["Int64", "Float64", "Int32", "Float32"]:
            numeric_cols.append(col)
    return numeric_cols


def _validate_columns(df: pl.DataFrame, col1: str, col2: str) -> bool:
    """列の存在と数値型をチェック"""
    if col1 not in df.columns or col2 not in df.columns:
        print("指定された列が存在しません")
        return False
    return True


def _get_operation() -> str:
    """演算の種類を取得"""
    operations = ["加算 (+)", "減算 (-)", "乗算 (*)", "除算 (/)"]
    choice = _get_user_choice("演算を選択:", operations)
    return ["+", "-", "*", "/"][choice - 1]


def _get_new_column_name(col1: str, col2: str, operation: str) -> str:
    """新しい列名を生成"""
    op_names = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    default_name = f"{col1}_{op_names[operation]}_{col2}"

    custom_name = input(f"新しい列名 (デフォルト: {default_name}): ").strip()
    return custom_name if custom_name else default_name


def _prepare_numeric_data(df: pl.DataFrame, col1: str, col2: str) -> tuple[np.ndarray, np.ndarray]:
    """数値データを準備"""
    data1 = df[col1].to_numpy()
    data2 = df[col2].to_numpy()
    return data1, data2


def _check_division_by_zero(data2: np.ndarray) -> bool:
    """ゼロ除算をチェック"""
    return np.any(data2 == 0)


def _perform_calculation(data1: np.ndarray, data2: np.ndarray, operation: str) -> np.ndarray:
    """計算を実行"""
    if operation == "+":
        return data1 + data2
    elif operation == "-":
        return data1 - data2
    elif operation == "*":
        return data1 * data2
    elif operation == "/":
        return np.divide(data1, data2, out=np.zeros_like(data1), where=data2 != 0)


def _show_statistics(result: np.ndarray, new_col_name: str) -> None:
    """結果の統計情報を表示"""
    print(f"  新しい列 '{new_col_name}' の統計:")
    print(f"    平均: {np.mean(result):.4f}")
    print(f"    標準偏差: {np.std(result):.4f}")
    print(f"    最小値: {np.min(result):.4f}")
    print(f"    最大値: {np.max(result):.4f}")


def arithmetic_columns(df: pl.DataFrame, config: PreprocessingConfig = None) -> pl.DataFrame:
    """列同士の四則演算"""
    numeric_cols = _get_column_names(df)

    if len(numeric_cols) < 2:
        print("四則演算には最低2つの数値列が必要です")
        return df

    print(f"\n利用可能な数値列 ({len(numeric_cols)}個):")
    for i, col in enumerate(numeric_cols, 1):
        print(f"  {i}. {col}")

    try:
        col1_idx = int(input("1つ目の列番号: ").strip()) - 1
        col2_idx = int(input("2つ目の列番号: ").strip()) - 1

        if not (0 <= col1_idx < len(numeric_cols) and 0 <= col2_idx < len(numeric_cols)):
            print("無効な列番号です")
            return df

        col1, col2 = numeric_cols[col1_idx], numeric_cols[col2_idx]

        if not _validate_columns(df, col1, col2):
            return df

        operation = _get_operation()
        new_col_name = _get_new_column_name(col1, col2, operation)

        # 設定を記録
        if config is not None:
            if not hasattr(config, "arithmetic_config"):
                config.arithmetic_config = []
            config.arithmetic_config.append([col1, col2, operation, new_col_name])

        data1, data2 = _prepare_numeric_data(df, col1, col2)

        if operation == "/" and _check_division_by_zero(data2):
            print("⚠️ 警告: ゼロ除算が発生する可能性があります。ゼロの場合は0で置換します。")

        result = _perform_calculation(data1, data2, operation)

        df_pd = df.to_pandas()
        df_pd[new_col_name] = result

        _show_statistics(result, new_col_name)

        return pl.from_pandas(df_pd)

    except ValueError:
        print("無効な入力です")
        return df


def execute_arithmetic_operations(df: pl.DataFrame, config: PreprocessingConfig = None) -> pl.DataFrame:
    """四則演算の実行ループ"""
    print("\n=== 四則演算による特徴量生成 ===")

    while True:
        continue_options = [
            "四則演算を実行",
            "四則演算を終了",
        ]

        choice = _get_user_choice("四則演算メニュー:", continue_options)

        if choice == 1:
            df = arithmetic_columns(df, config)
            print(f"現在のデータ形状: {df.shape}")
        else:
            break

    return df


def execute_preprocessing_loop(df, config: PreprocessingConfig = None) -> pl.DataFrame:
    """前処理のメインループ"""
    print("\n=== 前処理メニュー ===")
    print(f"現在のデータ形状: {df.shape}")

    # pandas.DataFrameの場合はpolars.DataFrameに変換
    if hasattr(df, "to_pandas"):  # polars.DataFrameの場合
        processed_df = df.clone()
    else:  # pandas.DataFrameの場合
        import polars as pl

        processed_df = pl.from_pandas(df)

    while True:
        preprocessing_options = [
            "カテゴリカル変数エンコーディング",
            "多項式特徴量生成",
            "四則演算による特徴量生成",
            "前処理を終了",
        ]

        try:
            choice = _get_user_choice("前処理を選択してください:", preprocessing_options)

            if choice == 1:
                processed_df = unified_encoding(processed_df, config)
            elif choice == 2:
                processed_df = polynomial_features(processed_df, config)
            elif choice == 3:
                processed_df = execute_arithmetic_operations(processed_df, config)
            elif choice == 4:
                print("前処理を終了します")
                break
            else:
                print("1~4の範囲で入力してください")
                continue

            print(f"\n現在のデータ形状: {processed_df.shape}")

        except ValueError:
            print("数値を入力してください")
            continue
        except KeyboardInterrupt:
            print("\n\n前処理を中断しました")
            break

    return processed_df


def apply_saved_preprocessing(df: pl.DataFrame, config: PreprocessingConfig) -> pl.DataFrame:
    """保存された前処理設定を適用"""
    print("\n=== 保存された前処理設定を適用 ===")

    processed_df = df.clone()

    # エンコーディング設定を適用
    if config.encoding_config:
        print("カテゴリカル変数エンコーディングを適用...")
        for col, encoding_type in config.encoding_config.items():
            if col in processed_df.columns and encoding_type != "skip":
                processed_df = _apply_encoding(processed_df, col, encoding_type)

    # 多項式特徴量設定を適用
    if config.polynomial_config:
        print("多項式特徴量生成を適用...")
        numeric_cols = [
            col
            for col in processed_df.columns
            if str(processed_df[col].dtype) in ["Int64", "Float64", "Int32", "Float32"]
        ]
        if numeric_cols:
            processed_df = _generate_polynomial_features(
                processed_df,
                numeric_cols,
                config.polynomial_config.get("degree", 2),
                config.polynomial_config.get("interaction_only", False),
            )

    # 四則演算設定を適用
    if config.arithmetic_config:
        print("四則演算による特徴量生成を適用...")
        for operation_config in config.arithmetic_config:
            col1, col2, operation, new_col = operation_config
            if col1 in processed_df.columns and col2 in processed_df.columns:
                data1 = processed_df[col1].to_numpy()
                data2 = processed_df[col2].to_numpy()
                result = _perform_calculation(data1, data2, operation)
                df_pd = processed_df.to_pandas()
                df_pd[new_col] = result
                processed_df = pl.from_pandas(df_pd)

    print(f"前処理適用完了: {df.shape} → {processed_df.shape}")
    return processed_df


def save_preprocessing_config(config: PreprocessingConfig, target_column: str, feature_columns: list[str]):
    """前処理設定を保存"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"preprocessing_config_{target_column}_{timestamp}.pkl"

    config.feature_names = feature_columns
    config.save(filename)
    print(f"✓ 前処理設定を保存しました: {filename}")
    return filename


def load_preprocessing_config(filepath: str) -> PreprocessingConfig:
    """前処理設定を読み込み"""
    config = PreprocessingConfig()
    config.load(filepath)
    print(f"✓ 前処理設定を読み込みました: {filepath}")
    return config
