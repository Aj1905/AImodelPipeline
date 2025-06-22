from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .session_manager import tuning_session
from .utils import _get_user_choice


def _get_basic_cv_config(cv_method: str) -> dict[str, Any]:
    """基本的なCV設定を取得"""
    config: dict[str, Any] = {"cv_method": cv_method}

    if cv_method == "3fold":
        config["cv_folds"] = 3
        print("✓ 3-fold CVを選択しました")
    elif cv_method == "kfold":
        config["cv_folds"] = _get_kfold_folds()
        print(f"✓ {config['cv_folds']}-fold CVを選択しました")

    return config


def _get_kfold_folds() -> int:
    """K-fold CVの分割数を取得"""
    while True:
        try:
            cv_folds = int(input("K-fold CVの分割数 (3-10推奨): ").strip())
            if 2 <= cv_folds <= 20:
                return cv_folds
            else:
                print("2~20の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_train_size_config() -> dict[str, Any]:
    """学習データサイズの設定を取得"""
    print("\n学習データサイズの設定方法を選択してください:")
    train_size_options = [
        "割合指定 - データ全体に対する割合で指定 (例: 20 = 20%)",
        "絶対数指定 - 具体的な行数で指定 (例: 1000 = 1000行)",
    ]

    train_size_choice = _get_user_choice("学習データサイズの設定方法:", train_size_options)
    config: dict[str, Any] = {}

    if train_size_choice == 1:
        config["train_size_ratio"] = _get_train_size_ratio()
        config["train_size_type"] = "ratio"
    else:
        config["train_size_absolute"] = _get_train_size_absolute()
        config["train_size_type"] = "absolute"

    return config


def _get_train_size_ratio() -> float:
    """学習データサイズの割合を取得"""
    while True:
        try:
            train_size_ratio = int(input("学習データサイズ (データ数の割合、例: 20 = 20%): ").strip())
            if 5 <= train_size_ratio <= 80:
                return train_size_ratio / 100.0
            else:
                print("5~80の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_train_size_absolute() -> int:
    """学習データサイズの絶対数を取得"""
    while True:
        try:
            train_size_absolute = int(input("学習データサイズ (行数): ").strip())
            if 10 <= train_size_absolute <= 100000:
                return train_size_absolute
            else:
                print("10~100000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_val_size_config() -> dict[str, Any]:
    """検証データサイズの設定を取得"""
    print("\n検証データサイズの設定:")
    val_size_options = [
        "固定サイズ - 各foldで同じサイズの検証データ",
        "残り全て - 学習データ以降の全てのデータ",
    ]

    val_size_choice = _get_user_choice("検証データサイズの設定方法:", val_size_options)
    config: dict[str, Any] = {}

    if val_size_choice == 1:
        config["val_size"] = _get_val_size_fixed()
        config["val_size_type"] = "fixed"
    else:
        config["val_size_type"] = "remaining"

    return config


def _get_val_size_fixed() -> int:
    """固定検証データサイズを取得"""
    while True:
        try:
            val_size = int(input("検証データサイズ (行数): ").strip())
            if 1 <= val_size <= 10000:
                return val_size
            else:
                print("1~10000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_gap_config() -> dict[str, Any]:
    """Gapの設定を取得"""
    print("\nGapの設定:")
    print("Gapは学習データと検証データの間の時間的な距離を表します")
    print("データリークを防ぐために重要です")

    gap_options = [
        "Gapなし - 学習データの直後に検証データ",
        "Gapあり - 学習データと検証データの間に間隔を設ける",
    ]

    gap_choice = _get_user_choice("Gapの設定:", gap_options)
    config: dict[str, Any] = {}

    if gap_choice == 1:
        config["gap"] = 0
        print("✓ Gapなしを選択しました")
    else:
        config["gap"] = _get_gap_size()

    return config


def _get_gap_size() -> int:
    """Gapサイズを取得"""
    while True:
        try:
            gap_size = int(input("Gapサイズ (行数): ").strip())
            if 0 <= gap_size <= 1000:
                print(f"✓ Gapサイズ: {gap_size}行")
                return gap_size
            else:
                print("0~1000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_step_size_config() -> dict[str, Any]:
    """ステップサイズの設定を取得"""
    print("\nステップサイズの設定:")
    print("ステップサイズは次のfoldに進む際の移動量を表します")

    step_options = [
        "検証データサイズと同じ - 検証データ分だけ移動",
        "固定サイズ - 指定した行数だけ移動",
    ]

    step_choice = _get_user_choice("ステップサイズの設定方法:", step_options)
    config: dict[str, Any] = {}

    if step_choice == 1:
        config["step_size_type"] = "val_size"
        print("✓ ステップサイズ: 検証データサイズと同じ")
    else:
        config["step_size"] = _get_step_size_fixed()
        config["step_size_type"] = "fixed"

    return config


def _get_step_size_fixed() -> int:
    """固定ステップサイズを取得"""
    while True:
        try:
            step_size = int(input("ステップサイズ (行数): ").strip())
            if 1 <= step_size <= 10000:
                print(f"✓ ステップサイズ: {step_size}行")
                return step_size
            else:
                print("1~10000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_cv_method_config() -> dict[str, Any]:
    """クロスバリデーション方法の設定を取得"""
    print("\n=== クロスバリデーション方法の選択 ===")

    cv_methods = [
        "3-fold CV - 3分割交差検証 (推奨: 小〜中規模データ)",
        "K-fold CV - K分割交差検証 (推奨: 中〜大規模データ)",
        "ローリングウィンドウCV - 時系列データ用 (時間順序を保持)",
        "拡張ウィンドウCV - 時系列データ用 (履歴データを蓄積)",
    ]

    method_choice = _get_user_choice("クロスバリデーション方法を選択してください:", cv_methods)
    method_map = ["3fold", "kfold", "rolling", "expanding"]
    cv_method = method_map[method_choice - 1]

    # 基本的なCV設定を取得
    config = _get_basic_cv_config(cv_method)

    # 時系列CVの場合は追加設定を取得
    if cv_method in ["rolling", "expanding"]:
        print(f"\n=== {cv_method.upper()} CV設定 ===")

        # 学習データサイズ設定
        train_config = _get_train_size_config()
        config.update(train_config)

        # 検証データサイズ設定
        val_config = _get_val_size_config()
        config.update(val_config)

        # Gap設定
        gap_config = _get_gap_config()
        config.update(gap_config)

        # ステップサイズ設定
        step_config = _get_step_size_config()
        config.update(step_config)

    return config


def _calculate_cv_parameters(data: pd.DataFrame, cv_config: dict[str, Any]) -> tuple[int, int, int, int | None]:
    """CVパラメータを計算"""
    n_samples = len(data)

    # 学習データサイズの計算
    if cv_config["train_size_type"] == "ratio":
        train_size = int(n_samples * cv_config["train_size_ratio"])
    else:
        train_size = cv_config["train_size_absolute"]

    # Gapサイズの取得
    gap_size = cv_config.get("gap", 0)

    # 検証データサイズの取得
    if cv_config["val_size_type"] == "fixed":
        val_size = cv_config["val_size"]
    else:
        val_size = None

    # ステップサイズの計算
    if cv_config["step_size_type"] == "val_size":
        step_size = val_size if val_size else train_size // 5
    else:
        step_size = cv_config["step_size"]

    return train_size, gap_size, step_size, val_size


def _create_rolling_window_splits(
    n_samples: int, train_size: int, gap_size: int, step_size: int, val_size: int | None
) -> list[tuple[list[int], list[int]]]:
    """ローリングウィンドウCVの分割を作成"""
    splits = []
    start_idx = 0

    while True:
        train_end = start_idx + train_size
        val_start = train_end + gap_size

        if val_size is not None:
            val_end = val_start + val_size
        else:
            val_end = n_samples

        if train_end >= n_samples or val_start >= n_samples:
            break

        if val_end > val_start:
            train_indices = list(range(start_idx, train_end))
            val_indices = list(range(val_start, val_end))
            splits.append((train_indices, val_indices))

        start_idx += step_size

        if start_idx >= n_samples:
            break

    return splits


def _create_expanding_window_splits(
    n_samples: int, train_size: int, gap_size: int, step_size: int, val_size: int | None
) -> list[tuple[list[int], list[int]]]:
    """拡張ウィンドウCVの分割を作成"""
    splits = []
    start_idx = 0

    while True:
        train_end = start_idx + train_size
        val_start = train_end + gap_size

        if val_size is not None:
            val_end = val_start + val_size
        else:
            val_end = n_samples

        if train_end >= n_samples or val_start >= n_samples:
            break

        if val_end > val_start:
            train_indices = list(range(0, train_end))
            val_indices = list(range(val_start, val_end))
            splits.append((train_indices, val_indices))

        start_idx += step_size

        if start_idx >= n_samples:
            break

    return splits


def _print_cv_split_info(
    n_samples: int, train_size: int, gap_size: int, step_size: int, splits: list[tuple[list[int], list[int]]]
) -> None:
    """CV分割情報を表示"""
    print("\n=== 時系列CV分割情報 ===")
    print(f"データ総数: {n_samples}行")
    print(f"学習データサイズ: {train_size}行")
    print(f"Gap: {gap_size}行")
    print(f"ステップサイズ: {step_size}行")
    print(f"生成されたfold数: {len(splits)}")

    if len(splits) > 0:
        print(f"最初のfold: 学習{len(splits[0][0])}行, 検証{len(splits[0][1])}行")
        print(f"最後のfold: 学習{len(splits[-1][0])}行, 検証{len(splits[-1][1])}行")

    if len(splits) == 0:
        print("⚠️ 警告: 指定された設定では有効なfoldが生成されませんでした")
        print("設定を見直してください(学習データサイズ、Gap、ステップサイズなど)")


def _create_time_series_cv_splits(data: pd.DataFrame, cv_config: dict[str, Any]) -> list[tuple[list[int], list[int]]]:
    """時系列クロスバリデーション用の分割を作成"""
    n_samples = len(data)
    cv_method = cv_config["cv_method"]

    # CVパラメータを計算
    train_size, gap_size, step_size, val_size = _calculate_cv_parameters(data, cv_config)

    # CV方法に応じて分割を作成
    if cv_method == "rolling":
        splits = _create_rolling_window_splits(n_samples, train_size, gap_size, step_size, val_size)
    elif cv_method == "expanding":
        splits = _create_expanding_window_splits(n_samples, train_size, gap_size, step_size, val_size)
    else:
        splits = []

    # 分割情報を表示
    _print_cv_split_info(n_samples, train_size, gap_size, step_size, splits)

    return splits


def _suggest_parameters(trial, param_ranges: dict) -> tuple[dict, int]:
    """Optunaでパラメータを提案"""
    params = {}
    num_boost_round = 100

    for param_name, param_range in param_ranges.items():
        if param_name == "num_boost_round":
            num_boost_round = trial.suggest_int(
                param_name, param_range["low"], param_range["high"], log=param_range.get("log", False)
            )
        elif param_range["type"] == "int":
            params[param_name] = trial.suggest_int(
                param_name, param_range["low"], param_range["high"], log=param_range.get("log", False)
            )
        elif param_range["type"] == "float":
            params[param_name] = trial.suggest_float(
                param_name, param_range["low"], param_range["high"], log=param_range.get("log", False)
            )
        elif param_range["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_range["choices"])

    return params, num_boost_round


def _evaluate_kfold_cv(train_data, target_column, feature_columns, cv_config, params, num_boost_round) -> list[float]:
    """K-fold CVで評価"""
    from src.ml import LightGBMRegressor

    cv_folds = cv_config["cv_folds"]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(train_data):
        train_fold = train_data[train_idx]
        val_fold = train_data[val_idx]

        model = LightGBMRegressor(num_boost_round=num_boost_round, params=params, verbose_eval=True)
        model.train(train_fold.select(feature_columns), train_fold[target_column])

        y_pred = model.predict(val_fold.select(feature_columns))
        y_true = val_fold[target_column]

        score = r2_score(y_true.to_numpy(), y_pred.to_numpy())
        cv_scores.append(score)

    return cv_scores


def _evaluate_time_series_cv(
    train_data, target_column, feature_columns, cv_config, params, num_boost_round
) -> list[float]:
    """時系列CVで評価"""
    from src.ml import LightGBMRegressor

    train_data_pd = train_data.to_pandas()
    cv_splits = _create_time_series_cv_splits(train_data_pd, cv_config)
    cv_scores = []

    for train_indices, val_indices in cv_splits:
        train_fold = train_data[train_indices]
        val_fold = train_data[val_indices]

        model = LightGBMRegressor(num_boost_round=num_boost_round, params=params, verbose_eval=True)
        model.train(train_fold.select(feature_columns), train_fold[target_column])

        y_pred = model.predict(val_fold.select(feature_columns))
        y_true = val_fold[target_column]

        score = r2_score(y_true.to_numpy(), y_pred.to_numpy())
        cv_scores.append(score)

    return cv_scores


def _create_optuna_objective_with_cv(train_data, target_column, feature_columns, cv_config, param_ranges):
    """CV方法を指定したOptuna用の目的関数を作成"""

    def objective(trial):
        try:
            # パラメータを提案
            params, num_boost_round = _suggest_parameters(trial, param_ranges)
            params["verbose"] = 1

            cv_method = cv_config["cv_method"]

            # CV方法に応じて評価
            if cv_method in ["3fold", "kfold"]:
                cv_scores = _evaluate_kfold_cv(
                    train_data, target_column, feature_columns, cv_config, params, num_boost_round
                )
            elif cv_method in ["rolling", "expanding"]:
                cv_scores = _evaluate_time_series_cv(
                    train_data, target_column, feature_columns, cv_config, params, num_boost_round
                )
            else:
                cv_scores = []

            return np.mean(cv_scores) if cv_scores else -999.0

        except Exception as e:
            print(f"CV評価中にエラーが発生しました: {e}")
            return -999.0

    return objective


def _get_optuna_config() -> dict[str, Any]:
    """Optuna設定を取得"""
    while True:
        try:
            n_trials = int(input("Optuna試行回数 (50-200推奨): ").strip())
            if 10 <= n_trials <= 1000:
                return {"n_trials": n_trials}
            else:
                print("10~1000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_grid_search_config() -> dict[str, Any]:
    """Grid Search設定を取得"""
    print("⚠️ 注意: Grid Searchは時間がかかる可能性があります")
    while True:
        try:
            max_combinations = int(input("最大組み合わせ数 (1000以下推奨): ").strip())
            if 1 <= max_combinations <= 10000:
                return {"max_combinations": max_combinations}
            else:
                print("1~10000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_random_search_config() -> dict[str, Any]:
    """Random Search設定を取得"""
    while True:
        try:
            n_iter = int(input("Random Search試行回数 (50-200推奨): ").strip())
            if 10 <= n_iter <= 1000:
                return {"n_iter": n_iter}
            else:
                print("10~1000の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _get_tuning_config() -> dict[str, Any]:
    """ハイパーパラメータチューニングの設定を取得"""
    print("\n=== ハイパーパラメータチューニング設定 ===")

    cv_config = _get_cv_method_config()

    tuning_methods = [
        "Optuna (推奨) - ベイズ最適化による効率的な探索",
        "Grid Search - 全組み合わせ探索 (時間がかかる)",
        "Random Search - ランダム探索",
    ]

    method_choice = _get_user_choice("チューニング方法を選択してください:", tuning_methods)
    method_map = ["optuna", "grid", "random"]
    tuning_method = method_map[method_choice - 1]

    config = {
        "tuning_method": tuning_method,
        "cv_config": cv_config,
    }

    # チューニング方法に応じた設定を取得
    if tuning_method == "optuna":
        config.update(_get_optuna_config())
    elif tuning_method == "grid":
        config.update(_get_grid_search_config())
    elif tuning_method == "random":
        config.update(_get_random_search_config())

    return config


def _get_lightgbm_param_ranges() -> dict[str, dict[str, Any]]:
    """LightGBMのパラメータ範囲を定義"""
    return {
        "num_boost_round": {"type": "int", "low": 50, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "num_leaves": {"type": "int", "low": 10, "high": 200, "log": True},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
        "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
        "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
        "lambda_l1": {"type": "float", "low": 0.0, "high": 10.0, "log": True},
        "lambda_l2": {"type": "float", "low": 0.0, "high": 10.0, "log": True},
        "random_state": {"type": "int", "low": 42, "high": 42},
    }


def _optuna_tuning(train_data, target_column, feature_columns, tuning_config):
    """Optunaによるハイパーパラメータチューニング"""
    print("\n=== Optunaチューニング開始 ===")

    cv_config = tuning_config["cv_config"]
    n_trials = tuning_config["n_trials"]
    param_ranges = _get_lightgbm_param_ranges()

    objective = _create_optuna_objective_with_cv(train_data, target_column, feature_columns, cv_config, param_ranges)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())

    print(f"Optunaチューニング実行中... (試行回数: {n_trials})")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print("\n=== Optunaチューニング完了 ===")
    print(f"最良スコア: {best_score:.4f}")
    print(f"最良パラメータ: {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "strategy": f"Optuna ({cv_config['cv_method']} CV)",
        "cv_config": cv_config,
    }


def execute_hyperparameter_tuning(train_data, target_column, feature_columns):
    """ハイパーパラメータチューニングを実行"""
    print("\n=== ハイパーパラメータチューニング ===")

    tuning_config = _get_tuning_config()

    if tuning_config["tuning_method"] == "optuna":
        results = _optuna_tuning(train_data, target_column, feature_columns, tuning_config)
    else:
        print("Grid SearchとRandom Searchは現在サポートされていません")
        return None

    if not results:
        print("❌ チューニングに失敗しました")
        return None

    enhanced_results = tuning_session.add_results(results, target_column, feature_columns)

    return enhanced_results


def execute_staged_tuning(train_data, target_column, feature_columns):
    """段階的ハイパーパラメータチューニング"""
    print("\n=== 段階的ハイパーパラメータチューニング ===")
    print("第1段階: 粗い探索でパラメータ範囲を絞り込み")
    print("第2段階: 絞り込んだ範囲で詳細探索")

    stage1_config = _get_tuning_config()
    stage1_config["n_trials"] = min(stage1_config.get("n_trials", 50), 50)

    print("\n--- 第1段階: 粗い探索 ---")
    stage1_results = _optuna_tuning(train_data, target_column, feature_columns, stage1_config)

    if not stage1_results:
        print("❌ 第1段階のチューニングに失敗しました")
        return None

    print(f"第1段階完了 - 最良スコア: {stage1_results['best_score']:.4f}")

    continue_choice = input("\n第2段階に進みますか? (y/n): ").strip().lower()
    if continue_choice != "y":
        print("✓ 第1段階の結果を採用")
        final_results = stage1_results
    else:
        print("\n--- 第2段階: 詳細探索 ---")
        print("第2段階は現在実装されていません。第1段階の結果を採用します。")
        final_results = stage1_results
        print("✓ 第1段階の結果を採用")

    enhanced_results = tuning_session.add_results(final_results, target_column, feature_columns)

    return enhanced_results
