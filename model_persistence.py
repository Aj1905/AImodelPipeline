import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import polars as pl

# 相対インポートを絶対インポートに変更
try:
    from .utils import _get_user_choice
except ImportError:
    # 直接実行時のための絶対インポート
    # 現在のディレクトリをパスに追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from utils import _get_user_choice


def save_tuning_results_to_joblib(results: dict, target_column: str, feature_columns: list[str]) -> str | None:
    """チューニング結果をjoblibファイルに保存"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuning_results_{target_column}_{timestamp}.joblib"

        save_data = {
            "results": results,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "timestamp": timestamp,
            "feature_count": len(feature_columns),
        }

        joblib.dump(save_data, filename)
        print(f"✓ チューニング結果を保存しました: {filename}")
        return filename

    except Exception as e:
        print(f"❌ 保存中にエラーが発生しました: {e}")
        return None


def load_tuning_results_from_joblib(filename: str) -> dict | None:
    """joblibファイルからチューニング結果を読み込み"""
    try:
        if not os.path.exists(filename):
            print(f"❌ ファイルが見つかりません: {filename}")
            return None

        data = joblib.load(filename)

        print(f"✓ チューニング結果を読み込みました: {filename}")
        print(f"  保存日時: {data.get('timestamp', '不明')}")
        print(f"  目的変数: {data.get('target_column', '不明')}")
        print(f"  特徴量数: {data.get('feature_count', '不明')}")

        return data

    except Exception as e:
        print(f"❌ 読み込み中にエラーが発生しました: {e}")
        return None


def list_saved_tuning_results() -> list[str]:
    """保存されたチューニング結果ファイルを一覧表示"""
    pattern = "tuning_results_*.joblib"
    files = list(Path(".").glob(pattern))

    if not files:
        print("保存されたチューニング結果が見つかりません")
        return []

    print(f"\n保存されたチューニング結果 ({len(files)}個):")

    file_info = []
    for i, file_path in enumerate(files, 1):
        try:
            data = joblib.load(file_path)
            timestamp = data.get("timestamp", "不明")
            target = data.get("target_column", "不明")
            feature_count = data.get("feature_count", "不明")
            strategy = data.get("results", {}).get("strategy", "不明")
            best_score = data.get("results", {}).get("best_score", "不明")

            print(f"  {i}. {file_path.name}")
            print(f"     保存日時: {timestamp}")
            print(f"     目的変数: {target}")
            print(f"     特徴量数: {feature_count}")
            print(f"     手法: {strategy}")
            print(f"     最良スコア: {best_score}")
            print()

            file_info.append(str(file_path))

        except Exception as e:
            print(f"  {i}. {file_path.name} (読み込みエラー: {e})")

    return file_info


def interactive_tuning_result_loader():
    """対話的にチューニング結果を読み込み"""
    files = list_saved_tuning_results()

    if not files:
        return None

    try:
        choice = int(input("読み込むファイル番号を入力してください: ").strip())
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
            return load_tuning_results_from_joblib(selected_file)
        else:
            print(f"1~{len(files)}の範囲で入力してください")
            return None
    except ValueError:
        print("数値を入力してください")
        return None


def save_model_to_file(model: Any, target_column: str, feature_columns: list[str]) -> str | None:
    """モデルをファイルに保存"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{target_column}_{timestamp}.joblib"

        save_data = {
            "model": model,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "timestamp": timestamp,
            "feature_count": len(feature_columns),
        }

        joblib.dump(save_data, filename)
        print(f"✓ モデルを保存しました: {filename}")
        return filename

    except Exception as e:
        print(f"❌ モデル保存中にエラーが発生しました: {e}")
        return None


def load_model_from_file(filename: str) -> dict | None:
    """ファイルからモデルを読み込み"""
    try:
        if not os.path.exists(filename):
            print(f"❌ ファイルが見つかりません: {filename}")
            return None

        data = joblib.load(filename)

        print(f"✓ モデルを読み込みました: {filename}")
        print(f"  保存日時: {data.get('timestamp', '不明')}")
        print(f"  目的変数: {data.get('target_column', '不明')}")

        # 実際のモデルの特徴量数を取得
        model = data.get("model")
        if hasattr(model, "_feature_names") and model._feature_names:
            feature_count = len(model._feature_names)
            feature_source = "モデル"
        else:
            feature_count = data.get("feature_count", "不明")
            feature_source = "メタデータ"

        print(f"  特徴量数: {feature_count} ({feature_source}から取得)")

        return data

    except Exception as e:
        print(f"❌ モデル読み込み中にエラーが発生しました: {e}")
        return None


def list_saved_models() -> list[str]:
    """保存されたモデルファイルを一覧表示"""
    pattern = "model_*.joblib"
    files = list(Path(".").glob(pattern))

    if not files:
        print("保存されたモデルが見つかりません")
        return []

    print(f"\n保存されたモデル ({len(files)}個):")

    file_info = []
    for i, file_path in enumerate(files, 1):
        try:
            data = joblib.load(file_path)
            timestamp = data.get("timestamp", "不明")
            target = data.get("target_column", "不明")

            # 実際のモデルの特徴量数を取得
            model = data.get("model")
            if hasattr(model, "_feature_names") and model._feature_names:
                feature_count = len(model._feature_names)
                feature_source = "モデル"
            else:
                feature_count = data.get("feature_count", "不明")
                feature_source = "メタデータ"

            print(f"  {i}. {file_path.name}")
            print(f"     保存日時: {timestamp}")
            print(f"     目的変数: {target}")
            print(f"     特徴量数: {feature_count} ({feature_source}から取得)")
            print()

            file_info.append(str(file_path))

        except Exception as e:
            print(f"  {i}. {file_path.name} (読み込みエラー: {e})")

    return file_info


def interactive_model_loader():
    """対話的にモデルを読み込み"""
    files = list_saved_models()

    if not files:
        return None

    try:
        choice = int(input("読み込むモデル番号を入力してください: ").strip())
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
            return load_model_from_file(selected_file)
        else:
            print(f"1~{len(files)}の範囲で入力してください")
            return None
    except ValueError:
        print("数値を入力してください")
        return None


def _get_feature_columns_from_model(model_data: dict) -> list[str]:
    """モデルから特徴量名を取得"""
    model = model_data["model"]

    # 実際のモデルの特徴量名を使用(保存されたメタデータではなく)
    if hasattr(model, "_feature_names") and model._feature_names:
        feature_columns = model._feature_names
        print(f"✓ モデルの実際の特徴量名を使用: {len(feature_columns)}個")
    else:
        # フォールバック: 保存されたメタデータを使用
        feature_columns = model_data["feature_columns"]
        print(f"⚠️ 保存されたメタデータの特徴量名を使用: {len(feature_columns)}個")

    return feature_columns


def _input_feature_individual(feature_columns: list[str]) -> dict[str, float] | None:
    """個別入力で特徴量を取得"""
    print("\n各特徴量の値を入力してください:")
    feature_values = {}

    for feature in feature_columns:
        while True:
            try:
                value = input(f"{feature}: ").strip()
                if value.lower() in ["quit", "exit", "q"]:
                    print("予測をキャンセルしました")
                    return None

                # 数値に変換
                numeric_value = float(value)
                feature_values[feature] = numeric_value
                break
            except ValueError:
                print("❌ 数値を入力してください (例: 1.5, 10, -2.3)")

    return feature_values


def _input_feature_bulk(feature_columns: list[str]) -> dict[str, float] | None:
    """一括入力で特徴量を取得"""
    print(f"\n{len(feature_columns)}個の特徴量をカンマ区切りで入力してください:")
    print(f"順序: {', '.join(feature_columns)}")
    print("例: 1.5,2.3,0.8,10.2")

    while True:
        try:
            values_input = input("値: ").strip()
            if values_input.lower() in ["quit", "exit", "q"]:
                print("予測をキャンセルしました")
                return None

            values = [v.strip() for v in values_input.split(",")]

            if len(values) != len(feature_columns):
                print(f"❌ {len(feature_columns)}個の値を入力してください")
                continue

            # 各値を数値に変換
            feature_values = {}
            for feature, value in zip(feature_columns, values, strict=False):
                try:
                    feature_values[feature] = float(value)
                except ValueError:
                    print(f"❌ {feature}の値 '{value}' を数値に変換できません")
                    break
            else:
                return feature_values  # すべての値が正常に変換された場合

        except Exception as e:
            print(f"❌ 入力エラー: {e}")


def _input_feature_from_csv(feature_columns: list[str]) -> dict[str, float] | None:
    """CSVファイルから特徴量を取得"""
    print("\nCSVファイルから特徴量を読み込みます")
    print("CSVファイルは以下の形式である必要があります:")
    print("- ヘッダー行に特徴量名")
    print("- データ行に値(カンマ区切り)")
    print("例: sample_features.csv")

    filename = input("CSVファイル名: ").strip()

    try:
        if not os.path.exists(filename):
            print(f"❌ ファイルが見つかりません: {filename}")
            return None

        csv_data = pd.read_csv(filename)

        if len(csv_data) == 0:
            print("❌ CSVファイルにデータがありません")
            return None

        # 最初の行を使用
        row = csv_data.iloc[0]

        # 必要な特徴量がすべて含まれているかチェック
        missing_features = [f for f in feature_columns if f not in csv_data.columns]
        if missing_features:
            print(f"❌ 以下の特徴量がCSVファイルに含まれていません: {missing_features}")
            print(f"必要な特徴量: {feature_columns}")
            print(f"CSVファイルの列: {list(csv_data.columns)}")
            return None

        # 特徴量の値を取得
        feature_values = {}
        for feature in feature_columns:
            feature_values[feature] = float(row[feature])

        print(f"✓ CSVファイルから {len(feature_values)} 個の特徴量を読み込みました")
        return feature_values

    except Exception as e:
        print(f"❌ CSVファイル読み込み中にエラーが発生しました: {e}")
        return None


def _get_feature_values(feature_columns: list[str]) -> dict[str, float] | None:
    """特徴量の値を取得"""
    print("\n入力方法を選択してください:")
    input_options = [
        "個別入力(一つずつ入力)",
        "一括入力(カンマ区切り)",
        "CSVファイルから読み込み",
    ]

    input_choice = _get_user_choice("入力方法:", input_options)

    if input_choice == 1:
        return _input_feature_individual(feature_columns)
    elif input_choice == 2:
        return _input_feature_bulk(feature_columns)
    elif input_choice == 3:
        return _input_feature_from_csv(feature_columns)
    else:
        print("❌ 無効な選択です")
        return None


def predict_with_manual_input(model_data: dict) -> dict | None:
    """手動入力された特徴量で予測を実行"""
    try:
        model = model_data["model"]
        target_column = model_data["target_column"]

        # 特徴量名を取得
        feature_columns = _get_feature_columns_from_model(model_data)

        print("\n=== 手動特徴量入力による予測 ===")
        print(f"目的変数: {target_column}")
        print(f"必要な特徴量 ({len(feature_columns)}個):")

        # 特徴量の説明を表示
        for i, feature in enumerate(feature_columns, 1):
            print(f"  {i}. {feature}")

        print("\n💡 ヒント:")
        print("- 数値以外の入力は無効です")
        print("- 予測をキャンセルするには 'quit', 'exit', 'q' と入力してください")
        print("- 小数点は '.' を使用してください(例: 1.5)")

        # 特徴量の値を取得
        feature_values = _get_feature_values(feature_columns)
        if feature_values is None:
            return None

        # 入力された特徴量をDataFrameに変換
        input_data = pd.DataFrame([feature_values])

        # 特徴量の順序をモデルの期待する順序に合わせる
        input_data = input_data[feature_columns]

        print("\n入力された特徴量:")
        for feature in feature_columns:
            print(f"  {feature}: {feature_values[feature]}")

        # 予測実行
        print("\n予測実行中...")
        prediction = model.predict(input_data)[0]

        result = {
            "prediction": prediction,
            "feature_values": feature_values,
            "feature_columns": feature_columns,
            "target_column": target_column,
        }

        print("✓ 予測完了!")
        print(f"予測値 ({target_column}): {prediction}")

        return result

    except Exception as e:
        print(f"❌ 予測中にエラーが発生しました: {e}")
        return None


def predict_with_loaded_model(model_data: dict, test_data: pl.DataFrame) -> dict | None:
    """読み込んだモデルで予測を実行"""
    try:
        model = model_data["model"]
        target_column = model_data["target_column"]

        # 実際のモデルの特徴量名を使用(保存されたメタデータではなく)
        if hasattr(model, "_feature_names") and model._feature_names:
            feature_columns = model._feature_names
            print(f"✓ モデルの実際の特徴量名を使用: {len(feature_columns)}個")
        else:
            # フォールバック: 保存されたメタデータを使用
            feature_columns = model_data["feature_columns"]
            print(f"⚠️ 保存されたメタデータの特徴量名を使用: {len(feature_columns)}個")

        print(f"予測実行中... (特徴量: {len(feature_columns)}個)")

        # デバッグ情報を追加
        print(f"テストデータの形状: {test_data.shape}")
        print(f"テストデータの列: {list(test_data.columns)}")
        print(f"必要な特徴量: {feature_columns}")

        # 特徴量の存在確認
        missing_features = [col for col in feature_columns if col not in test_data.columns]
        if missing_features:
            print(f"❌ 必要な特徴量が不足しています: {missing_features}")
            print(f"利用可能な列: {list(test_data.columns)}")
            return None

        # 特徴量のデータ型確認
        print("特徴量のデータ型:")
        for col in feature_columns:
            dtype = test_data[col].dtype
            print(f"  {col}: {dtype}")

        # 欠損値の確認
        print("特徴量の欠損値:")
        for col in feature_columns:
            null_count = test_data[col].null_count()
            print(f"  {col}: {null_count}個")

        predictions = model.predict(test_data.select(feature_columns))

        result = {
            "predictions": predictions,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "prediction_count": len(predictions),
        }

        print(f"✓ 予測完了: {len(predictions)}件")
        return result

    except Exception as e:
        print(f"❌ 予測中にエラーが発生しました: {e}")
        import traceback

        print(f"詳細エラー: {traceback.format_exc()}")
        return None


def export_model_predictions(
    predictions: dict, test_data: pl.DataFrame, output_filename: str | None = None
) -> str | None:
    """予測結果をCSVファイルにエクスポート"""
    try:
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_column = predictions["target_column"]
            output_filename = f"predictions_{target_column}_{timestamp}.csv"

        test_data_pd = test_data.to_pandas()
        test_data_pd[f"{predictions['target_column']}_predicted"] = predictions["predictions"]

        test_data_pd.to_csv(output_filename, index=False)

        print(f"✓ 予測結果をエクスポートしました: {output_filename}")
        print(f"  データ数: {len(test_data_pd)}行")
        print(f"  列数: {len(test_data_pd.columns)}列")

        return output_filename

    except Exception as e:
        print(f"❌ エクスポート中にエラーが発生しました: {e}")
        return None


if __name__ == "__main__":
    print("=== モデル永続化モジュール ===")
    print("このファイルは直接実行するためのものではありません。")
    print("他のモジュールからインポートして使用してください。")
    print("\n利用可能な関数:")
    print("- save_tuning_results_to_joblib()")
    print("- load_tuning_results_from_joblib()")
    print("- list_saved_tuning_results()")
    print("- interactive_tuning_result_loader()")
    print("- save_model_to_file()")
    print("- load_model_from_file()")
    print("- list_saved_models()")
    print("- interactive_model_loader()")
    print("- predict_with_manual_input()")
    print("- predict_with_loaded_model()")
    print("- export_model_predictions()")
