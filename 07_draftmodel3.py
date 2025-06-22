import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクトパス設定後にインポート
from src.ml.modele.data_loader import load_and_join_tables, select_tables_interactively
from src.ml.modele.hyperparameter_tuner import execute_hyperparameter_tuning, execute_staged_tuning
from src.ml.modele.model_evaluator import _get_sampling_method_config, execute_model_evaluation
from src.ml.modele.model_persistence import (
    export_model_predictions,
    interactive_model_loader,
    interactive_tuning_result_loader,
    predict_with_loaded_model,
    predict_with_manual_input,
    save_model_to_file,
    save_tuning_results_to_joblib,
)
from src.ml.modele.preprocessor import (
    PreprocessingConfig,
    execute_preprocessing_loop,
    save_preprocessing_config,
)
from src.ml.modele.session_manager import tuning_session
from src.ml.modele.utils import _get_user_choice, _select_target_and_features

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

# sqliteパス(定数として定義)
DB_PATH = "/Users/aj/Documents/forecasting_poc/data/database.sqlite"

# テストデータの割合 (20%)
TEST_SIZE = 0.2

# 前処理設定
PREPROCESSING_CONFIG = {
    "transform_numeric": True,  # 数値特徴量の変換
    "impute_missing": True,  # 欠損値補完
    "unified_encoding": True,  # カテゴリカル変数のエンコーディング
    "polynomial_features": True,  # 多項式特徴量の生成
}


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


def handle_data_preprocessing():
    """データ読み込み・前処理を実行"""
    print("\n=== データ読み込み・前処理 ===")
    tables = select_tables_interactively(DB_PATH)
    data = load_and_join_tables(DB_PATH, tables)
    print(f"読み込み完了: {data.shape}")

    # 前処理設定を作成
    preprocessing_config = PreprocessingConfig()
    processed_data = execute_preprocessing_loop(data, preprocessing_config)
    print(f"前処理完了: {processed_data.shape}")

    target_column, feature_columns = _select_target_and_features(processed_data)

    # 前処理設定を保存
    save_choice = input("前処理設定を保存しますか? (y/n): ").strip().lower()
    if save_choice == "y":
        config_filename = save_preprocessing_config(preprocessing_config, target_column, feature_columns)
        tuning_session.preprocessing_config_file = config_filename

    tuning_session.current_data = processed_data
    tuning_session.target_column = target_column
    tuning_session.feature_columns = feature_columns

    print("✓ データ準備完了")
    print(f"  目的変数: {target_column}")
    print(f"  特徴量数: {len(feature_columns)}")


def handle_hyperparameter_tuning():
    """ハイパーパラメータチューニングを実行"""
    if not hasattr(tuning_session, "current_data"):
        print("❌ 先にデータ読み込み・前処理を実行してください")
        return

    results = execute_hyperparameter_tuning(
        tuning_session.current_data, tuning_session.target_column, tuning_session.feature_columns
    )

    if results:
        session_id = tuning_session.add_results(results, tuning_session.target_column, tuning_session.feature_columns)
        print(f"✓ チューニング結果をセッション {session_id} に保存しました")

        save_choice = input("結果をファイルに保存しますか? (y/n): ").strip().lower()
        if save_choice == "y":
            filename = save_tuning_results_to_joblib(
                results, tuning_session.target_column, tuning_session.feature_columns
            )
            if filename:
                print(f"✓ ファイルに保存しました: {filename}")


def handle_staged_tuning():
    """段階的チューニングを実行"""
    if not hasattr(tuning_session, "current_data"):
        print("❌ 先にデータ読み込み・前処理を実行してください")
        return

    results = execute_staged_tuning(
        tuning_session.current_data, tuning_session.target_column, tuning_session.feature_columns
    )

    if results:
        session_id = tuning_session.add_results(results, tuning_session.target_column, tuning_session.feature_columns)
        print(f"✓ 段階的チューニング結果をセッション {session_id} に保存しました")


def handle_model_evaluation():
    """モデル評価を実行"""
    if not hasattr(tuning_session, "current_data"):
        print("❌ 先にデータ読み込み・前処理を実行してください")
        return

    print("\n=== モデル評価 ===")
    print("既存のチューニング結果から選択してモデル評価を実行します")
    print("チューニング結果がない場合は、デフォルトパラメータまたは手動入力で評価を続行できます")

    sampling_config = _get_sampling_method_config()

    evaluation_results = execute_model_evaluation(
        tuning_session.current_data,
        tuning_session.target_column,
        tuning_session.feature_columns,
        test_size=TEST_SIZE,
        sampling_config=sampling_config,
    )

    # モデル評価が成功し、モデルが含まれている場合
    if evaluation_results and "model" in evaluation_results and evaluation_results["model"] is not None:
        print("\n=== モデル保存オプション ===")
        save_choice = input("評価済みモデルをファイルに保存しますか? (y/n): ").strip().lower()

        if save_choice == "y":
            filename = save_model_to_file(
                evaluation_results["model"], tuning_session.target_column, tuning_session.feature_columns
            )
            if filename:
                print(f"✓ モデルを保存しました: {filename}")
            else:
                print("❌ モデルの保存に失敗しました")
        else:
            print("モデルの保存をスキップしました")
    elif evaluation_results:
        print("⚠️ モデル評価は完了しましたが、モデルオブジェクトが含まれていません")
        print("評価結果の詳細:")
        if "test_metrics" in evaluation_results:
            print(f"  テストデータ R²: {evaluation_results['test_metrics'].get('R2', 'N/A'):.4f}")
            print(f"  テストデータ RMSE: {evaluation_results['test_metrics'].get('RMSE', 'N/A'):.4f}")
        if "train_metrics" in evaluation_results:
            print(f"  学習データ R²: {evaluation_results['train_metrics'].get('R2', 'N/A'):.4f}")
            print(f"  学習データ RMSE: {evaluation_results['train_metrics'].get('RMSE', 'N/A'):.4f}")
    else:
        print("❌ モデル評価に失敗しました")
        print("ヒント:")
        print("  1. 先にハイパーパラメータチューニングを実行してください")
        print("  2. デフォルトパラメータを使用して評価を続行してください")
        print("  3. 手動でパラメータを入力してください")


def handle_load_tuning_results():
    """保存済みチューニング結果を読み込み"""
    loaded_data = interactive_tuning_result_loader()
    if loaded_data:
        print("✓ チューニング結果を読み込みました")
        tuning_session.target_column = loaded_data["target_column"]
        tuning_session.feature_columns = loaded_data["feature_columns"]


def handle_prediction_with_saved_model():
    """保存済みモデルでの予測を実行"""
    print("\n=== 保存済みモデルでの予測 ===")

    # モデルを読み込み
    model_data = interactive_model_loader()
    if not model_data:
        return

    # データベースからテーブルを選択
    print("\n予測に使用するテーブルを選択してください:")
    tables = select_tables_interactively(DB_PATH)
    if not tables:
        print("❌ テーブルが選択されませんでした")
        return

    # データを読み込み
    test_data = load_and_join_tables(DB_PATH, tables)
    print(f"✓ テストデータを読み込みました: {test_data.shape}")

    # 予測実行
    predictions = predict_with_loaded_model(model_data, test_data)

    if predictions:
        # 予測結果をCSVに出力
        export_choice = input("予測結果をCSVにエクスポートしますか? (y/n): ").strip().lower()
        if export_choice == "y":
            export_model_predictions(predictions, test_data)


def handle_prediction_with_current_data():
    """現在のデータでの予測を実行"""
    model_data = interactive_model_loader()
    if model_data:
        if hasattr(tuning_session, "current_data"):
            predictions = predict_with_loaded_model(model_data, tuning_session.current_data)
            if predictions:
                export_choice = input("予測結果をCSVにエクスポートしますか? (y/n): ").strip().lower()
                if export_choice == "y":
                    export_model_predictions(predictions, tuning_session.current_data)
        else:
            print("❌ 予測用のデータがありません。先にデータ読み込みを実行してください")


def handle_manual_prediction():
    """手動特徴量入力による予測を実行"""
    model_data = interactive_model_loader()
    if model_data:
        prediction_result = predict_with_manual_input(model_data)
        if prediction_result:
            print("\n=== 予測結果サマリー ===")
            print(f"目的変数: {prediction_result['target_column']}")
            print(f"予測値: {prediction_result['prediction']}")

            # 予測結果をファイルに保存するオプション
            save_choice = input("\n予測結果をファイルに保存しますか? (y/n): ").strip().lower()
            if save_choice == "y":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manual_prediction_{prediction_result['target_column']}_{timestamp}.txt"

                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write("=== 手動特徴量入力による予測結果 ===\n")
                        f.write(f"予測日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"目的変数: {prediction_result['target_column']}\n")
                        f.write(f"予測値: {prediction_result['prediction']}\n\n")
                        f.write("入力された特徴量:\n")
                        for feature, value in prediction_result["feature_values"].items():
                            f.write(f"  {feature}: {value}\n")

                    print(f"✓ 予測結果を保存しました: {filename}")
                except Exception as e:
                    print(f"❌ ファイル保存中にエラーが発生しました: {e}")


def load_test_data_from_file(target_column):
    """ファイルからテストデータを読み込み"""
    print("\nテストデータファイルを選択してください:")
    print("1. CSVファイル")
    print("2. Parquetファイル")
    print("3. 現在のデータセット(目的変数を除外)")

    file_choice = input("選択してください (1-3): ").strip()

    if file_choice == "1":
        file_path = input("CSVファイルのパスを入力してください: ").strip()
        try:
            test_data = pl.read_csv(file_path)
            print(f"✓ CSVファイルを読み込みました: {test_data.shape}")
            return test_data
        except Exception as e:
            print(f"❌ CSVファイルの読み込みに失敗しました: {e}")
            return None

    elif file_choice == "2":
        file_path = input("Parquetファイルのパスを入力してください: ").strip()
        try:
            test_data = pl.read_parquet(file_path)
            print(f"✓ Parquetファイルを読み込みました: {test_data.shape}")
            return test_data
        except Exception as e:
            print(f"❌ Parquetファイルの読み込みに失敗しました: {e}")
            return None

    elif file_choice == "3":
        if hasattr(tuning_session, "current_data"):
            # 現在のデータから目的変数を除外
            if target_column in tuning_session.current_data.columns:
                test_data = tuning_session.current_data.drop(target_column)
                print(f"✓ 現在のデータセットから目的変数を除外しました: {test_data.shape}")
            else:
                test_data = tuning_session.current_data
                print(f"✓ 現在のデータセットを使用します: {test_data.shape}")
            return test_data
        else:
            print("❌ 現在のデータセットがありません。先にデータ読み込みを実行してください")
            return None
    else:
        print("❌ 無効な選択です")
        return None


def save_prediction_results(predictions, test_data, target_column):
    """予測結果をCSVに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{target_column}_{timestamp}.csv"

    try:
        # テストデータと予測結果を結合
        test_data_pd = test_data.to_pandas()
        test_data_pd[f"{target_column}_predicted"] = predictions["predictions"]

        # CSVファイルに保存
        test_data_pd.to_csv(output_filename, index=False, encoding="utf-8")

        print(f"✓ 予測結果をCSVに出力しました: {output_filename}")
        print(f"  データ数: {len(test_data_pd)}行")
        print(f"  列数: {len(test_data_pd.columns)}列")
        print(f"  予測列: {target_column}_predicted")

        # 予測結果の統計情報を表示
        pred_values = predictions["predictions"]
        print("\n=== 予測結果の統計 ===")
        print(f"  平均: {pred_values.mean():.4f}")
        print(f"  最小: {pred_values.min():.4f}")
        print(f"  最大: {pred_values.max():.4f}")
        print(f"  標準偏差: {pred_values.std():.4f}")

    except Exception as e:
        print(f"❌ CSV出力中にエラーが発生しました: {e}")
        import traceback

        print(f"詳細エラー: {traceback.format_exc()}")


def handle_prediction_without_target():
    """目的変数なしテストデータでの予測を実行"""
    print("\n=== 目的変数なしテストデータでの予測 ===")

    # モデルを読み込み
    model_data = interactive_model_loader()
    if not model_data:
        return

    # テストデータを読み込み
    test_data = load_test_data_from_file(model_data["target_column"])
    if test_data is None or test_data.is_empty():
        print("❌ テストデータが空です")
        return

    # 特徴量の確認
    feature_columns = model_data["feature_columns"]
    missing_features = [col for col in feature_columns if col not in test_data.columns]

    if missing_features:
        print(f"❌ 必要な特徴量が不足しています: {missing_features}")
        print(f"利用可能な列: {list(test_data.columns)}")
        return

    # 予測実行
    print(f"\n予測実行中... (特徴量: {len(feature_columns)}個)")
    predictions = predict_with_loaded_model(model_data, test_data)

    if predictions:
        target_column = model_data["target_column"]
        save_prediction_results(predictions, test_data, target_column)


def display_session_details(session_id, session_data):
    """セッションの詳細を表示"""
    print(f"\n=== セッション{session_id}の詳細 ===")
    print(f"実行時刻: {session_data.get('timestamp', '不明')}")
    print(f"目的変数: {session_data.get('target_column', '不明')}")
    print(f"特徴量数: {len(session_data.get('feature_columns', []))}")
    print(f"手法: {session_data.get('strategy', session_data.get('method', '不明'))}")
    print(f"最良スコア: {session_data.get('best_score', '不明')}")

    if "best_params" in session_data:
        print("\n最適パラメータ:")
        for key, value in session_data["best_params"].items():
            print(f"  {key}: {value}")

    # 評価結果がある場合
    if "test_metrics" in session_data:
        print("\n評価結果:")
        print(f"  テストデータ R²: {session_data['test_metrics'].get('R2', 'N/A'):.4f}")
        print(f"  テストデータ RMSE: {session_data['test_metrics'].get('RMSE', 'N/A'):.4f}")
    if "train_metrics" in session_data:
        print(f"  学習データ R²: {session_data['train_metrics'].get('R2', 'N/A'):.4f}")
        print(f"  学習データ RMSE: {session_data['train_metrics'].get('RMSE', 'N/A'):.4f}")


def _handle_specific_session_detail(history):
    """特定のセッションの詳細を表示"""
    try:
        session_id = int(input("セッションIDを入力してください: "))
        if session_id in history:
            session_data = history[session_id]
            display_session_details(session_id, session_data)
        else:
            print(f"❌ セッション{session_id}が見つかりません")
    except ValueError:
        print("❌ 無効なセッションIDです")


def _handle_tuning_results_only(history):
    """チューニング結果のみを表示"""
    print("\n=== チューニング結果のみ ===")
    tuning_sessions = [(sid, data) for sid, data in history.items() if "best_params" in data]
    if tuning_sessions:
        for session_id, session_data in tuning_sessions:
            print(
                f"セッション{session_id}: {session_data.get('target_column', '不明')} - スコア: {session_data.get('best_score', '不明')}"
            )
    else:
        print("チューニング結果がありません")


def _handle_evaluation_results_only(history):
    """評価結果のみを表示"""
    print("\n=== 評価結果のみ ===")
    eval_sessions = [(sid, data) for sid, data in history.items() if "test_metrics" in data or "train_metrics" in data]
    if eval_sessions:
        for session_id, session_data in eval_sessions:
            print(f"セッション{session_id}: {session_data.get('target_column', '不明')}")
            if "test_metrics" in session_data:
                print(f"  テスト R²: {session_data['test_metrics'].get('R2', 'N/A'):.4f}")
    else:
        print("評価結果がありません")


def _handle_latest_results(history):
    """最新の結果を表示"""
    latest_results = tuning_session.get_latest_results()
    if latest_results:
        print(f"\n=== 最新の結果 (セッション{max(history.keys())}) ===")
        print(f"実行時刻: {latest_results.get('timestamp', '不明')}")
        print(f"目的変数: {latest_results.get('target_column', '不明')}")
        print(f"手法: {latest_results.get('strategy', latest_results.get('method', '不明'))}")
        if "best_score" in latest_results:
            print(f"最良スコア: {latest_results['best_score']}")
    else:
        print("❌ 最新の結果が見つかりません")


def handle_session_detail_options(history):
    """セッション履歴の詳細表示オプションを処理"""
    print("\n=== 詳細表示オプション ===")
    detail_options = [
        "特定のセッションの詳細を表示",
        "チューニング結果のみを表示",
        "評価結果のみを表示",
        "最新の結果を表示",
        "戻る",
    ]

    detail_choice = _get_user_choice("選択してください:", detail_options)

    if detail_choice == 1:
        _handle_specific_session_detail(history)
    elif detail_choice == 2:
        _handle_tuning_results_only(history)
    elif detail_choice == 3:
        _handle_evaluation_results_only(history)
    elif detail_choice == 4:
        _handle_latest_results(history)
    elif detail_choice == 5:
        return


def handle_session_history():
    """セッション履歴を表示"""
    print("\n=== セッション履歴表示 ===")
    print("過去のチューニング結果と評価結果を表示します")

    history = tuning_session.get_session_history()
    if not history:
        print("❌ セッション履歴がありません")
        print("ヒント: ハイパーパラメータチューニングまたはモデル評価を実行すると履歴が作成されます")
        return

    tuning_session.list_session_history()
    handle_session_detail_options(history)


def _handle_data_operations(choice):
    """データ関連の操作を処理"""
    if choice == 1:
        handle_data_preprocessing()
    elif choice == 5:
        handle_load_tuning_results()
    else:
        return False
    return True


def _handle_model_operations(choice):
    """モデル関連の操作を処理"""
    if choice == 2:
        handle_hyperparameter_tuning()
    elif choice == 3:
        handle_staged_tuning()
    elif choice == 4:
        handle_model_evaluation()
    else:
        return False
    return True


def _handle_prediction_operations(choice):
    """予測関連の操作を処理"""
    if choice == 6:
        handle_prediction_with_saved_model()
    elif choice == 7:
        handle_prediction_with_current_data()
    elif choice == 8:
        handle_manual_prediction()
    elif choice == 9:
        handle_prediction_without_target()
    else:
        return False
    return True


def handle_menu_choice(choice):
    """メニュー選択を処理"""
    # データ関連の操作
    if _handle_data_operations(choice):
        return True

    # モデル関連の操作
    if _handle_model_operations(choice):
        return True

    # 予測関連の操作
    if _handle_prediction_operations(choice):
        return True

    # その他の操作
    if choice == 10:
        handle_session_history()
    elif choice == 11:
        return False  # 終了フラグ
    else:
        print("無効な選択です")

    return True  # 継続フラグ


def main():
    """メイン関数"""
    print("=" * 80)
    print("🤖 機械学習パイプライン (リファクタリング版)")
    print("=" * 80)

    # データベースパスをセッションに設定
    tuning_session.set_db_path(DB_PATH)

    while True:
        print("\n=== メインメニュー ===")
        options = [
            "データ読み込み・前処理",
            "ハイパーパラメータチューニング",
            "段階的チューニング",
            "モデル評価 (既存チューニング結果から選択)",
            "保存済みチューニング結果の読み込み",
            "保存済みモデルでの予測",
            "手動特徴量入力による予測",
            "目的変数なしテストデータでの予測",
            "セッション履歴表示",
            "終了",
        ]

        choice = _get_user_choice("実行する処理を選択してください:", options)

        if not handle_menu_choice(choice):
            print("プログラムを終了します")
            break


if __name__ == "__main__":
    main()
