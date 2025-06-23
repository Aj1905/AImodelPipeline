import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.implementations.lightgbm_model import LightGBMRegressor
from src.features.managers.feature_manager import FeatureManager
from src.features.managers.target_manager import TargetManager
from src.pipelines.implementations.tree_pipeline import TreeModelPipeline
from src.data.utils.data_loader import (
    load_data_from_sqlite_polars,
    validate_db_path,
)
from src.data.utils.interactive_selector import (
    get_table_info_summary,
    select_table_interactively,
    validate_table_exists,
)


@dataclass
class Args:
    model_path: str
    db_path: str = "data/database.sqlite"
    table: str = None
    output_path: str = "predictions.csv"
    config_path: str = None


def feature_engineering(data: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """特徴量エンジニアリングを実行(学習時と同じ処理)"""
    processed_data = data.clone()

    # Convert date string to datetime if date column exists
    if "date" in data.columns:
        processed_data = processed_data.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
        )

        # Day of week: 0=Monday, 1=Tuesday, ..., 6=Sunday
        processed_data = processed_data.with_columns(pl.col("datetime").dt.weekday().alias("dow"))

    # 給料日が 25 日以降なので、25 日以降を月末として特徴量にする
    if "date_day" in data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("date_day") >= 25).then(1).otherwise(0).alias("is_month_end")
        )

    # 週末
    if "dow" in processed_data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("dow") >= 4).then(1).otherwise(0).alias("is_weekend")
        )

    # ランチタイム・ディナータイム
    if "time" in data.columns:
        processed_data = processed_data.with_columns(
            pl.when(pl.col("time").is_in([11, 12, 13])).then(1).otherwise(0).alias("is_lunch")
        )
        processed_data = processed_data.with_columns(
            pl.when(pl.col("time") >= 18).then(1).otherwise(0).alias("is_dinner")
        )

    # 指定された特徴量列のみを選択
    available_columns = [col for col in feature_columns if col in processed_data.columns]
    processed_data = processed_data.select(available_columns)

    print("特徴量の確認:")
    print(processed_data.head())
    print(f"使用する特徴量列: {available_columns}")

    return processed_data


def load_saved_model(model_path: Path) -> tuple[TreeModelPipeline, dict]:
    """保存されたモデルと設定を読み込む"""
    print(f"📥 モデルを読み込み中: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    # パイプラインを初期化
    model = LightGBMRegressor()
    feature_manager = FeatureManager()
    target_manager = TargetManager(pl.Series("dummy", [0.0]))
    pipeline = TreeModelPipeline(model=model, feature_manager=feature_manager, target_manager=target_manager)

    # パイプライン全体を読み込み
    pipeline.load_model(model_path)

    # 学習情報を表示
    model = pipeline.get_model()
    if hasattr(model, "print_training_info"):
        model.print_training_info()

    # 設定ファイルも読み込み
    config_path = model_path.with_suffix(".json")
    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        print(f"📋 設定情報を読み込みました: {config_path}")
    else:
        print("⚠️  設定ファイルが見つかりません。デフォルト設定を使用します。")

    return pipeline, config


def interactive_setup(db_path: Path) -> str:
    """対話的にテーブルを設定"""
    print("🔍 データベース設定")
    print("=" * 40)

    # テーブル選択
    table_name = select_table_interactively(db_path)
    if not table_name:
        print("テーブルが選択されませんでした。処理を終了します。")
        sys.exit(1)

    print(f"✅ 選択されたテーブル: {table_name}")

    # テーブル情報の表示
    table_info = get_table_info_summary(db_path, table_name)
    if table_info:
        print("\n📊 テーブル情報:")
        print(f"  行数: {table_info['row_count']:,}")
        print(f"  列数: {table_info['column_count']}")
        print("  列の詳細:")
        for col_name, col_info in table_info["columns"].items():
            if "error" not in col_info:
                print(
                    f"    {col_name}: {col_info['type']} (NULL: {col_info['null_count']}, ユニーク: {col_info['unique_count']})"
                )

    return table_name


def validate_features(test_data: pl.DataFrame, required_features: list[str]) -> list[str]:
    """テストデータに必要な特徴量が存在するかチェック"""
    available_features = []
    missing_features = []

    for feature in required_features:
        if feature in test_data.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    if missing_features:
        print(f"❌ 以下の特徴量が不足しています: {missing_features}")
        print(f"利用可能な列: {list(test_data.columns)}")
        return []

    return available_features


def _parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="保存されたモデルを使用して予測を実行")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="保存されたモデルファイルのパス",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/database.sqlite",
        help="SQLiteデータベースファイルのパス (デフォルト: data/database.sqlite)",
    )
    parser.add_argument(
        "--table",
        type=str,
        help="使用するテーブル名 (指定しない場合は対話的に選択)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="predictions.csv",
        help="予測結果の出力パス (デフォルト: predictions.csv)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="設定ファイルのパス (指定しない場合はモデルファイルと同じ場所の.jsonファイル)",
    )
    return parser.parse_args(namespace=Args)


def _load_model_and_config(args):
    """モデルと設定を読み込む"""
    model_path = Path(args.model_path)
    try:
        pipeline, config = load_saved_model(model_path)
        print("✅ モデルの読み込みが完了しました")
        return pipeline, config
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        sys.exit(1)


def _setup_database_and_table(args, db_path):
    """データベースとテーブルの設定"""
    # テーブル選択(コマンドライン引数で指定されていない場合)
    if not args.table:
        table_name = interactive_setup(db_path)
        args.table = table_name
    else:
        # コマンドライン引数で指定された場合の検証
        if not validate_table_exists(db_path, args.table):
            print(f"❌ 指定されたテーブル '{args.table}' が存在しません")
            sys.exit(1)


def _load_and_validate_data(args, config):
    """データを読み込み、検証する"""
    # テストデータを読み込み
    try:
        print("\n📥 テストデータ読み込み中...")
        test_data = load_data_from_sqlite_polars(args.db_path, args.table)
        print(f"✅ テストデータ読み込み完了: {test_data.shape}")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        sys.exit(1)

    # データの基本情報表示
    print("\n📊 テストデータの基本情報:")
    print(f"  形状: {test_data.shape}")
    print(f"  列名: {test_data.columns}")

    # 設定から特徴量列を取得
    feature_columns = config.get("feature_columns", [])
    if not feature_columns:
        print("❌ 設定ファイルに特徴量列の情報がありません")
        sys.exit(1)

    print(f"  必要な特徴量列: {feature_columns}")

    # 特徴量の存在確認
    available_features = validate_features(test_data, feature_columns)
    if not available_features:
        print("❌ 必要な特徴量が不足しています")
        sys.exit(1)

    return test_data, available_features


def _execute_prediction(pipeline, test_data, available_features):
    """予測を実行する"""
    # 特徴量エンジニアリング
    print("\n🔧 特徴量エンジニアリング実行中...")
    engineered_data = feature_engineering(test_data, available_features)

    # 予測実行
    print("\n🔄 予測実行中...")
    try:
        model = pipeline.get_model()
        predictions = model.predict(engineered_data)
        print(f"✅ 予測完了: {len(predictions)}件")
        return model, predictions
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        sys.exit(1)


def _save_and_display_results(args, test_data, predictions, model):
    """結果を保存し、表示する"""
    # 予測結果の表示
    print("\n📈 予測結果サマリー:")
    print(f"  予測件数: {len(predictions)}")
    print(f"  平均予測値: {predictions.mean():.4f}")
    print(f"  最小予測値: {predictions.min():.4f}")
    print(f"  最大予測値: {predictions.max():.4f}")
    print(f"  標準偏差: {predictions.std():.4f}")

    # 予測結果をテストデータと結合
    result_data = test_data.with_columns(predictions.alias("prediction"))

    # 最初の10件を表示
    print("\n📋 予測結果(最初の10件):")
    print(result_data.select(["prediction", *list(test_data.columns[:5])]).head(10))

    # 結果をCSVに保存
    try:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_data.write_csv(output_path)
        print(f"\n💾 予測結果を保存しました: {output_path}")
    except Exception as e:
        print(f"❌ 結果保存エラー: {e}")

    # 特徴量重要度を表示(モデルが対応している場合)
    try:
        importance = model.get_feature_importance()
        print("\n🎯 特徴量重要度 (トップ10):")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_importance[:10]:
            print(f"  {feature}: {imp:.4f}")
    except Exception as e:
        print(f"⚠️  特徴量重要度の取得に失敗しました: {e}")


def main():
    """メイン関数"""
    args = _parse_arguments()

    print("🚀 保存されたモデルを使用した予測実行")
    print("=" * 60)
    print(f"モデルパス: {args.model_path}")
    print(f"データベースパス: {args.db_path}")

    # モデルと設定を読み込み
    pipeline, config = _load_model_and_config(args)

    # データベースの存在確認
    db_path = Path(args.db_path)
    if not validate_db_path(db_path):
        sys.exit(1)

    # データベースとテーブルの設定
    _setup_database_and_table(args, db_path)

    # データを読み込み、検証
    test_data, available_features = _load_and_validate_data(args, config)

    # 予測を実行
    model, predictions = _execute_prediction(pipeline, test_data, available_features)

    # 結果を保存し、表示
    _save_and_display_results(args, test_data, predictions, model)

    print("\n✅ 予測処理が完了しました!")


if __name__ == "__main__":
    main()
