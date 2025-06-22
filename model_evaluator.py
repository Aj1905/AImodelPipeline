import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from .hyperparameter_tuner import _create_time_series_cv_splits
from .session_manager import tuning_session
from .utils import _get_user_choice, analyze_train_test_comparison


def calculate_metrics(y_true: pl.Series, y_pred: np.ndarray) -> dict:
    """評価指標を計算"""
    y_true_np = y_true.to_numpy()

    return {
        "MSE": mean_squared_error(y_true_np, y_pred),
        "MAE": mean_absolute_error(y_true_np, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true_np, y_pred)),
        "R2": r2_score(y_true_np, y_pred),
    }


def _get_sampling_method_config() -> dict:
    """データ分割方法の設定を取得"""
    print("\n=== テストデータ選択 ===")
    sampling_methods = [
        "ランダムサンプリング - 毎回異なるランダムな分割",
        "固定データ - 別のテーブルをテストデータとして指定",
    ]
    method_choice = _get_user_choice("データ分割方法を選択してください:", sampling_methods)

    if method_choice == 1:
        print("✓ ランダムサンプリングを選択しました")
        return {"method": "random", "random_state": None}
    else:
        print("✓ 固定データを選択しました")
        print("テストデータとして使用するテーブルを指定してください")

        # テストデータテーブルの選択
        from .data_loader import select_table_interactively

        db_path = tuning_session.db_path
        test_table = select_table_interactively(db_path)

        print(f"✓ テストデータテーブル: {test_table}")
        return {"method": "fixed", "random_state": 42, "test_table": test_table}


def split_train_test_data(
    data: pl.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    sampling_method: str = "fixed",
    random_state: int | None = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """データを学習データとテストデータに分割"""

    data_pd = data.to_pandas()

    feature_cols = [col for col in data_pd.columns if col != target_column]
    x = data_pd[feature_cols]
    y = data_pd[target_column]

    if sampling_method == "random":
        random_state = None
        print("ランダムサンプリングでデータを分割します")
    else:
        print(f"固定データ分割でデータを分割します (random_state={random_state})")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    train_data = pl.from_pandas(pd.concat([x_train, y_train], axis=1))
    test_data = pl.from_pandas(pd.concat([x_test, y_test], axis=1))

    print(f"学習データ: {train_data.shape}")
    print(f"テストデータ: {test_data.shape}")

    return train_data, test_data, y_train, y_test


def _ensure_feature_columns_consistency(
    train_data: pl.DataFrame, test_data: pl.DataFrame, feature_columns: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """学習データとテストデータの特徴量列の整合性を確保"""
    print("\n=== 特徴量列の整合性チェック ===")

    # 学習データに存在する特徴量列
    train_features = [col for col in feature_columns if col in train_data.columns]
    missing_in_train = [col for col in feature_columns if col not in train_data.columns]

    # テストデータに存在する特徴量列
    test_features = [col for col in feature_columns if col in test_data.columns]
    missing_in_test = [col for col in feature_columns if col not in test_data.columns]

    print(f"学習データの特徴量列: {len(train_features)}/{len(feature_columns)}")
    print(f"テストデータの特徴量列: {len(test_features)}/{len(feature_columns)}")

    if missing_in_train:
        print(f"⚠️ 学習データに不足している列: {missing_in_train}")

    if missing_in_test:
        print(f"⚠️ テストデータに不足している列: {missing_in_test}")

    # 両方のデータセットに存在する共通の特徴量列を取得
    common_features = list(set(train_features) & set(test_features))
    print(f"共通の特徴量列: {len(common_features)}個")

    if len(common_features) == 0:
        raise ValueError("学習データとテストデータに共通の特徴量列がありません")

    if len(common_features) < len(feature_columns):
        print(f"⚠️ 一部の特徴量列が使用できません。{len(common_features)}個の列でモデルを学習します。")
        print(f"使用する特徴量列: {common_features[:5]}{'...' if len(common_features) > 5 else ''}")

    # 共通の特徴量列のみを使用
    train_data_filtered = train_data.select(
        common_features + [col for col in train_data.columns if col not in common_features]
    )
    test_data_filtered = test_data.select(
        common_features + [col for col in test_data.columns if col not in common_features]
    )

    return train_data_filtered, test_data_filtered, common_features


def train_and_evaluate_model(
    train_data: pl.DataFrame, test_data: pl.DataFrame, target_column: str, feature_columns: list[str], best_params: dict
) -> dict:
    """モデルを学習・評価"""
    from src.ml import LightGBMRegressor

    params = best_params.copy()
    num_boost_round = params.pop("num_boost_round", 100)

    model = LightGBMRegressor(num_boost_round=num_boost_round, params=params)

    print("モデル学習中...")
    model.train(train_data.select(feature_columns), train_data[target_column])

    print("学習データでの予測...")
    train_pred = model.predict(train_data.select(feature_columns))
    train_true = train_data[target_column]

    train_metrics = calculate_metrics(train_true, train_pred)

    print("\n=== 評価結果 ===")
    print("学習データ:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # テストデータに目的変数が存在するかチェック
    test_metrics = None
    test_pred = None
    test_true = None

    if target_column in test_data.columns:
        print("テストデータでの予測...")
        test_pred = model.predict(test_data.select(feature_columns))
        test_true = test_data[target_column]
        test_metrics = calculate_metrics(test_true, test_pred)

        print("テストデータ:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("⚠️ テストデータに目的変数が存在しないため、テストデータでの評価をスキップします")
        print(f"  テストデータの列: {list(test_data.columns)}")
        print(f"  期待される目的変数: {target_column}")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_true": train_true,
        "test_true": test_true,
    }


def _get_default_parameters() -> dict:
    """デフォルトパラメータを取得"""
    return {
        "best_params": {
            "num_boost_round": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": 1,
        },
        "target_column": "default",
        "feature_columns": [],
        "best_score": "default",
        "timestamp": "default",
    }


def _handle_no_tuning_results() -> dict | None:
    """チューニング結果がない場合の処理"""
    print("⚠️ 保存されたチューニング結果がありません")
    print("以下のオプションから選択してください:")

    options = [
        "デフォルトパラメータを使用",
        "手動でパラメータを入力",
        "チューニングを先に実行",
    ]

    choice = _get_user_choice("選択してください:", options)

    if choice == 1:
        print("✓ デフォルトパラメータを使用します")
        return _get_default_parameters()
    elif choice == 2:
        print("✓ 手動でパラメータを入力します")
        return _get_manual_parameters()
    elif choice == 3:
        print("❌ 先にハイパーパラメータチューニングを実行してください")
        return None
    else:
        print("❌ 無効な選択です")
        return None


def _create_tuning_result_options(tuning_results: list) -> list[str]:
    """チューニング結果の選択肢を作成"""
    options = []
    for session_id, session_data in tuning_results:
        target_col = session_data.get("target_column", "不明")
        feature_count = len(session_data.get("feature_columns", []))
        best_score = session_data.get("best_score", "不明")
        timestamp = session_data.get("timestamp", "不明")

        option_text = f"セッション{session_id}: {target_col} (特徴量{feature_count}個, スコア{best_score}, {timestamp})"
        options.append(option_text)

    # 追加オプション
    options.append("最新のチューニング結果を使用")
    options.append("デフォルトパラメータを使用")
    options.append("手動でパラメータを入力")

    return options


def _handle_tuning_result_choice(choice: int, tuning_results: list) -> dict | None:
    """チューニング結果の選択を処理"""
    if choice <= len(tuning_results):
        # 選択されたチューニング結果を返す
        selected_session_id, selected_data = tuning_results[choice - 1]
        print(f"✓ セッション{selected_session_id}のチューニング結果を選択しました")
        return selected_data
    elif choice == len(tuning_results) + 1:
        # 最新の結果を使用
        latest_results = tuning_session.get_latest_results()
        if latest_results and "best_params" in latest_results:
            print("✓ 最新のチューニング結果を使用します")
            return latest_results
        else:
            print("❌ 最新のチューニング結果が見つかりません")
            return None
    elif choice == len(tuning_results) + 2:
        # デフォルトパラメータを使用
        print("✓ デフォルトパラメータを使用します")
        return _get_default_parameters()
    elif choice == len(tuning_results) + 3:
        # 手動でパラメータを入力
        print("✓ 手動でパラメータを入力します")
        return _get_manual_parameters()
    else:
        print("❌ 無効な選択です")
        return None


def _select_tuning_results() -> dict | None:
    """既存のチューニング結果を選択"""
    print("\n=== チューニング結果選択 ===")

    # セッション履歴からチューニング結果を取得
    session_history = tuning_session.get_session_history()

    # チューニング結果のみをフィルタリング
    tuning_results = []
    if session_history:
        for session_id, session_data in session_history.items():
            if "best_params" in session_data:
                tuning_results.append((session_id, session_data))

    if not tuning_results:
        return _handle_no_tuning_results()

    print(f"利用可能なチューニング結果: {len(tuning_results)}件")

    # 選択肢を作成
    options = _create_tuning_result_options(tuning_results)

    # ユーザーに選択させる
    choice = _get_user_choice("使用するチューニング結果を選択してください:", options)

    # 選択を処理
    return _handle_tuning_result_choice(choice, tuning_results)


def _get_manual_parameters() -> dict:
    """手動でパラメータを入力"""
    print("\n=== 手動パラメータ入力 ===")
    print("LightGBMのパラメータを手動で入力してください")
    print("(空の場合はデフォルト値を使用)")

    params = {}

    # 基本的なパラメータ
    param_definitions = {
        "num_boost_round": {"default": 100, "type": int, "description": "ブースティング回数"},
        "learning_rate": {"default": 0.1, "type": float, "description": "学習率"},
        "max_depth": {"default": 6, "type": int, "description": "最大深さ"},
        "num_leaves": {"default": 31, "type": int, "description": "葉の数"},
        "min_child_samples": {"default": 20, "type": int, "description": "最小サンプル数"},
        "subsample": {"default": 0.8, "type": float, "description": "サブサンプル率"},
        "colsample_bytree": {"default": 0.8, "type": float, "description": "特徴量サブサンプル率"},
        "random_state": {"default": 42, "type": int, "description": "乱数シード"},
        "verbose": {"default": 1, "type": int, "description": "詳細度"},
    }

    for param_name, param_info in param_definitions.items():
        while True:
            try:
                user_input = input(
                    f"{param_name} ({param_info['description']}) [デフォルト: {param_info['default']}]: "
                ).strip()

                if user_input == "":
                    params[param_name] = param_info["default"]
                    break
                else:
                    params[param_name] = param_info["type"](user_input)
                    break
            except ValueError:
                print(f"❌ 無効な値です。{param_info['type'].__name__}型で入力してください")

    print("✓ 手動パラメータを設定しました")
    return {
        "best_params": params,
        "target_column": "manual",
        "feature_columns": [],
        "best_score": "manual",
        "timestamp": "manual",
    }


def execute_model_evaluation(train_data, target_column, feature_columns, test_size=0.2, sampling_config=None):
    """モデル評価を実行"""
    print("\n=== モデル評価 ===")
    print(f"データサイズ: {train_data.shape}")
    print(f"目的変数: {target_column}")
    print(f"特徴量数: {len(feature_columns)}")

    # チューニング結果の選択
    selected_results = _select_tuning_results()
    if selected_results is None:
        print("❌ チューニング結果の選択に失敗しました")
        print("ヒント: 先にハイパーパラメータチューニングを実行するか、デフォルトパラメータを使用してください")
        return None

    # 選択されたチューニング結果からパラメータを取得
    if "best_params" in selected_results:
        best_params = selected_results["best_params"]
        print("\n✓ 選択されたチューニング結果のパラメータを使用:")
        print(f"  目的変数: {selected_results.get('target_column', '不明')}")
        print(f"  特徴量数: {len(selected_results.get('feature_columns', []))}")
        print(f"  ベストスコア: {selected_results.get('best_score', '不明')}")
        print(f"  タイムスタンプ: {selected_results.get('timestamp', '不明')}")

        # パラメータの詳細表示
        print("\n使用するパラメータ:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("❌ 選択された結果にパラメータが含まれていません")
        return None

    # 固定データが選択された場合の処理
    if sampling_config and sampling_config.get("method") == "fixed":
        print("\n✓ 固定データを選択しました")
        print(f"  現在のデータ: {train_data.shape}")
        print(f"  目的変数: {target_column}")
        print(f"  特徴量数: {len(feature_columns)}")
        if "test_table" in sampling_config:
            print(f"  テストデータテーブル: {sampling_config['test_table']}")
        print("   全データを学習に使用し、指定されたテーブルをテストデータとして使用します")

    evaluation_methods = [
        "学習・テストデータ分割評価",
        "クロスバリデーション評価",
    ]

    method_choice = _get_user_choice("評価方法を選択してください:", evaluation_methods)

    if method_choice == 1:
        return _execute_train_test_evaluation(
            train_data, target_column, feature_columns, test_size, sampling_config, best_params
        )
    elif method_choice == 2:
        return _execute_cross_validation_evaluation(train_data, target_column, feature_columns, best_params)
    else:
        print("❌ 無効な選択です")
        return None


def _execute_train_test_evaluation(train_data, target_column, feature_columns, test_size, sampling_config, best_params):
    """学習・テストデータ分割評価を実行"""
    print(f"✓ 選択されたパラメータを使用: {best_params}")

    sampling_config = sampling_config or _get_sampling_method_config()

    # 固定データが選択された場合の処理
    if sampling_config.get("method") == "fixed" and "test_table" in sampling_config:
        print("\n=== 固定データでのモデル評価 ===")
        print("全データを学習に使用し、指定されたテーブルをテストデータとして使用します")

        # テストデータテーブルを読み込み
        from .data_loader import load_and_join_tables

        db_path = tuning_session.db_path
        test_table = sampling_config["test_table"]

        print(f"テストデータテーブル '{test_table}' を読み込み中...")
        test_data = load_and_join_tables(db_path, [test_table])
        print(f"テストデータ読み込み完了: {test_data.shape}")

        # テストデータの前処理(学習データと同じ前処理を適用)
        from .preprocessor import apply_saved_preprocessing, execute_preprocessing_loop, load_preprocessing_config

        # 保存された前処理設定がある場合は使用
        if hasattr(tuning_session, "preprocessing_config_file") and tuning_session.preprocessing_config_file:
            try:
                print(f"保存された前処理設定を使用: {tuning_session.preprocessing_config_file}")
                config = load_preprocessing_config(tuning_session.preprocessing_config_file)
                processed_test_data = apply_saved_preprocessing(test_data, config)
            except Exception as e:
                print(f"⚠️ 保存された前処理設定の読み込みに失敗しました: {e}")
                print("新しく前処理を実行します")
                processed_test_data = execute_preprocessing_loop(test_data)
        else:
            print("保存された前処理設定がありません。新しく前処理を実行します")
            processed_test_data = execute_preprocessing_loop(test_data)

        test_data = processed_test_data
        print(f"テストデータ前処理完了: {test_data.shape}")

        # 全データを学習データとして使用
        train_data_full = train_data

        print(f"学習データ: {train_data_full.shape} (全データ)")
        print(f"テストデータ: {test_data.shape} (テーブル: {test_table})")

        # 特徴量列の整合性をチェック
        print("\n=== 特徴量列の整合性チェック ===")
        train_features = set(train_data_full.columns)
        test_features = set(test_data.columns)

        print(f"学習データの列数: {len(train_features)}")
        print(f"テストデータの列数: {len(test_features)}")

        # 学習データに存在するがテストデータに存在しない列
        missing_in_test = train_features - test_features
        if missing_in_test:
            print(
                f"⚠️ テストデータに不足している列: {list(missing_in_test)[:5]}{'...' if len(missing_in_test) > 5 else ''}"
            )

        # テストデータに存在するが学習データに存在しない列
        extra_in_test = test_features - train_features
        if extra_in_test:
            print(f"⚠️ テストデータの余分な列: {list(extra_in_test)[:5]}{'...' if len(extra_in_test) > 5 else ''}")

        evaluation_results = train_and_evaluate_model(
            train_data_full, test_data, target_column, feature_columns, best_params
        )

        # 結果に固定データ情報を追加
        evaluation_results["fixed_data"] = True
        evaluation_results["test_table"] = test_table
        evaluation_results["train_data_shape"] = train_data_full.shape
        evaluation_results["test_data_shape"] = test_data.shape

    else:
        # 通常の学習・テストデータ分割
        train_split, test_split, _, _ = split_train_test_data(
            train_data,
            target_column,
            test_size=test_size,
            sampling_method=sampling_config["method"],
            random_state=sampling_config["random_state"],
        )

        evaluation_results = train_and_evaluate_model(
            train_split, test_split, target_column, feature_columns, best_params
        )

    enhanced_results = tuning_session.add_results(evaluation_results, target_column, feature_columns)
    return enhanced_results


def _execute_cross_validation_evaluation(train_data, target_column, feature_columns, best_params):
    """クロスバリデーション評価を実行"""
    from .hyperparameter_tuner import _get_cv_method_config

    print(f"✓ 選択されたパラメータを使用: {best_params}")

    cv_config = _get_cv_method_config()

    print("\n=== クロスバリデーション実行 ===")

    try:
        from src.ml import LightGBMRegressor
    except ImportError:
        print("⚠️ LightGBMRegressorが見つかりません。ダミーの結果を返します。")
        return {
            "cv_method": cv_config["cv_method"],
            "cv_config": cv_config,
            "fold_details": [],
            "scores": [0.8, 0.75, 0.82],
            "mean_score": 0.79,
            "std_score": 0.03,
            "all_y_true": [],
            "all_y_pred": [],
            "strategy": f"CV Evaluation ({cv_config['cv_method']})",
        }

    params = best_params.copy()
    num_boost_round = params.pop("num_boost_round", 100)

    cv_method = cv_config["cv_method"]
    fold_details = []
    all_y_true = []
    all_y_pred = []
    scores = []

    if cv_method in ["3fold", "kfold"]:
        cv_folds = cv_config["cv_folds"]
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_data), 1):
            print(f"Fold {fold_idx}/{cv_folds} 実行中...")

            train_fold = train_data[train_idx]
            val_fold = train_data[val_idx]

            model = LightGBMRegressor(num_boost_round=num_boost_round, params=params)
            model.train(train_fold.select(feature_columns), train_fold[target_column])

            y_pred = model.predict(val_fold.select(feature_columns))
            y_true = val_fold[target_column]

            fold_score = r2_score(y_true.to_numpy(), y_pred.to_numpy())
            scores.append(fold_score)

            train_stats = {
                "mean": float(train_fold[target_column].mean()),
                "std": float(train_fold[target_column].std()),
            }
            test_stats = {
                "mean": float(val_fold[target_column].mean()),
                "std": float(val_fold[target_column].std()),
            }

            fold_details.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_fold),
                    "test_size": len(val_fold),
                    "score": fold_score,
                    "train_stats": train_stats,
                    "test_stats": test_stats,
                    "y_true": y_true.to_numpy(),
                    "y_pred": y_pred.to_numpy(),
                }
            )

            all_y_true.extend(y_true.to_numpy())
            all_y_pred.extend(y_pred.to_numpy())

    elif cv_method in ["rolling", "expanding"]:
        train_data_pd = train_data.to_pandas()
        cv_splits = _create_time_series_cv_splits(train_data_pd, cv_config)

        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits, 1):
            print(f"Fold {fold_idx}/{len(cv_splits)} 実行中...")

            train_fold = train_data[train_indices]
            val_fold = train_data[val_indices]

            model = LightGBMRegressor(num_boost_round=num_boost_round, params=params)
            model.train(train_fold.select(feature_columns), train_fold[target_column])

            y_pred = model.predict(val_fold.select(feature_columns))
            y_true = val_fold[target_column]

            fold_score = r2_score(y_true.to_numpy(), y_pred.to_numpy())
            scores.append(fold_score)

            train_stats = {
                "mean": float(train_fold[target_column].mean()),
                "std": float(train_fold[target_column].std()),
            }
            test_stats = {
                "mean": float(val_fold[target_column].mean()),
                "std": float(val_fold[target_column].std()),
            }

            fold_details.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_fold),
                    "test_size": len(val_fold),
                    "score": fold_score,
                    "train_stats": train_stats,
                    "test_stats": test_stats,
                    "y_true": y_true.to_numpy(),
                    "y_pred": y_pred.to_numpy(),
                }
            )

            all_y_true.extend(y_true.to_numpy())
            all_y_pred.extend(y_pred.to_numpy())

    print("\n=== クロスバリデーション結果 ===")
    print(f"平均スコア: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"各Foldのスコア: {[f'{s:.4f}' for s in scores]}")

    evaluation_results = {
        "cv_method": cv_method,
        "cv_config": cv_config,
        "fold_details": fold_details,
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "strategy": f"CV Evaluation ({cv_method})",
    }

    _analyze_train_test_comparison(fold_details, cv_method, all_y_true, all_y_pred, scores, cv_config)

    enhanced_results = tuning_session.add_results(evaluation_results, target_column, feature_columns)
    return enhanced_results


def _analyze_train_test_comparison(fold_details, cv_method, all_y_true, all_y_pred, scores, cv_config):
    """学習データとテストデータの比較分析"""
    analyze_train_test_comparison(fold_details, cv_method, all_y_true, all_y_pred, scores, cv_config)
