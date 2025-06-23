import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lightgbm_model import LightGBMRegressor
from src.features.managers.feature_manager import FeatureManager
from src.features.managers.target_manager import TargetManager
from src.pipelines.implementations.tree_pipeline import TreeModelPipeline
from src.data.utils.data_loader import (
    load_data_from_sqlite_polars,
    validate_db_path,
)
from src.data.utils.interactive_selector import (
    get_table_columns,
    interactive_setup,
    validate_table_exists,
)


@dataclass
class Args:
    db_path: str
    table: str = None
    target_column: str = None
    feature_columns: list[str] = None
    save_model: bool = True
    model_save_path: str = "trained_model/lightgbm_model.pkl"
    time_series_split: bool = False
    time_column: str = "date"
    # MLflow関連のオプション
    use_mlflow: bool = True
    experiment_name: str = "lightgbm_experiment"
    run_name: str = None
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    no_mlflow: bool = False


def _process_datetime_features(data: pl.DataFrame) -> pl.DataFrame:
    """日時関連の特徴量を処理する"""
    processed_data = data.clone()

    # Convert date string to datetime if date column exists
    if "date" in data.columns:
        processed_data = processed_data.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
        )

        # Day of week: 0=Monday, 1=Tuesday, ..., 6=Sunday
        processed_data = processed_data.with_columns(pl.col("datetime").dt.weekday().alias("dow"))

    return processed_data


def _process_time_features(data: pl.DataFrame) -> pl.DataFrame:
    """時間関連の特徴量を処理する"""
    processed_data = data.clone()

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

    return processed_data


def _convert_string_columns(data: pl.DataFrame, available_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """文字列型の列を数値に変換する"""
    processed_data = data.clone()
    updated_columns = available_columns.copy()

    print("\n🔍 データ型の確認:")
    for col in processed_data.columns:
        dtype = processed_data[col].dtype
        print(f"  {col}: {dtype}")

        # 文字列型の場合は数値に変換を試行
        if dtype == pl.Utf8:
            print(f"    ⚠️  文字列型の列 '{col}' を数値に変換します")
            try:
                # 空文字列をNaNに変換してから数値に変換
                processed_data = processed_data.with_columns(
                    pl.col(col).str.replace("", "null").cast(pl.Float64, strict=False)
                )
                print("    ✅ 変換成功")
            except Exception as e:
                print(f"    ❌ 変換失敗: {e}")
                # 変換できない場合は除外
                if col in updated_columns:
                    updated_columns.remove(col)
                    print(f"    🗑️  列 '{col}' を除外します")

    return processed_data, updated_columns


def _handle_missing_values(data: pl.DataFrame) -> pl.DataFrame:
    """欠損値を処理する"""
    processed_data = data.clone()

    print("\n🔧 欠損値の処理:")
    for col in processed_data.columns:
        null_count = processed_data[col].null_count()
        if null_count > 0:
            print(f"  {col}: {null_count}個の欠損値を0で補完")
            processed_data = processed_data.with_columns(pl.col(col).fill_null(0))

    return processed_data


def feature_engineering(
    data: pl.DataFrame, feature_columns: list[str], keep_date_for_split: bool = False
) -> pl.DataFrame:
    """特徴量エンジニアリングを実行"""
    # 日時関連の特徴量を処理
    processed_data = _process_datetime_features(data)

    # 時間関連の特徴量を処理
    processed_data = _process_time_features(processed_data)

    # 指定された特徴量列のみを選択
    available_columns = [col for col in feature_columns if col in processed_data.columns]

    # 時系列分割が必要な場合はdate列を保持(datetime列に変換されても元のdate列を保持)
    if keep_date_for_split and "date" in data.columns and "date" not in available_columns:
        available_columns.append("date")

    processed_data = processed_data.select(available_columns)

    # 文字列型の列を数値に変換
    processed_data, available_columns = _convert_string_columns(processed_data, available_columns)

    # 欠損値の処理
    processed_data = _handle_missing_values(processed_data)

    # 最終的な列選択
    final_columns = [col for col in available_columns if col in processed_data.columns]
    processed_data = processed_data.select(final_columns)

    print("\n📊 特徴量の確認:")
    print(processed_data.head())
    print(f"使用する特徴量列: {final_columns}")
    print("最終的なデータ型:")
    for col in processed_data.columns:
        print(f"  {col}: {processed_data[col].dtype}")

    return processed_data


def parse_and_validate_args() -> Args:
    """コマンドライン引数を解析し、必要に応じて対話的設定を行う"""
    parser = argparse.ArgumentParser(
        description="Interactive ML Model Training with SQLite and MLflow Integration",
        epilog="""
Examples:
  # 基本的な使用方法（MLflow有効）
  python src/scripts/07_concise_draftmodel.py

  # MLflowを無効にする場合
  python src/scripts/07_concise_draftmodel.py --no-mlflow

  # カスタム実験名を指定
  python src/scripts/07_concise_draftmodel.py --experiment-name "my_experiment"

  # MLflow UIで結果を確認
  python -m mlflow ui
  # ブラウザで http://localhost:5000 にアクセス
        """
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
        "--target-column",
        type=str,
        help="ターゲット列名 (指定しない場合は対話的に選択)",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="+",
        help="特徴量列名 (指定しない場合は対話的に選択)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="モデルを保存しない",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="trained_model/lightgbm_model.pkl",
        help="モデル保存パス (デフォルト: trained_model/lightgbm_model.pkl)",
    )
    parser.add_argument(
        "--time-series-split",
        action="store_true",
        help="時系列分割を実行する",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="date",
        help="時系列分割に使用する日付列名 (デフォルト: date)",
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="MLflowを使用する",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflowを使用しない",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lightgbm_experiment",
        help="MLflowの実験名 (デフォルト: lightgbm_experiment)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="MLflowの実行名 (指定しない場合は自動生成)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflowのトラッキングURI (デフォルト: sqlite:///mlflow.db)",
    )
    args = parser.parse_args(namespace=Args)

    # 時系列分割の設定をArgsに反映
    args.time_series_split = args.time_series_split
    args.time_column = args.time_column
    # MLflow関連の設定をArgsに反映
    args.use_mlflow = args.use_mlflow
    args.experiment_name = args.experiment_name
    args.run_name = args.run_name
    args.mlflow_tracking_uri = args.mlflow_tracking_uri
    
    # --no-mlflowオプションが指定された場合はMLflowを無効化
    if args.no_mlflow:
        args.use_mlflow = False

    print("🚀 Interactive ML Model Training with SQLite")
    print("=" * 60)
    print(f"データベースパス: {args.db_path}")

    db_path = Path(args.db_path)
    if not validate_db_path(db_path):
        sys.exit(1)

    # 対話的設定(コマンドライン引数で指定されていない場合)
    if not args.table or not args.target_column or not args.feature_columns:
        try:
            table_name, target_column, feature_columns = interactive_setup(db_path)
            args.table = table_name
            args.target_column = target_column
            args.feature_columns = feature_columns
        except ValueError as e:
            print(f"❌ 設定エラー: {e}")
            sys.exit(1)
    else:
        # コマンドライン引数で指定された場合の検証
        if not validate_table_exists(db_path, args.table):
            print(f"❌ 指定されたテーブル '{args.table}' が存在しません")
            sys.exit(1)

        all_columns = [col[1] for col in get_table_columns(db_path, args.table)]
        if args.target_column not in all_columns:
            print(f"❌ 指定されたターゲット列 '{args.target_column}' が存在しません")
            sys.exit(1)

        missing_columns = [col for col in args.feature_columns if col not in all_columns]
        if missing_columns:
            print(f"❌ 以下の特徴量列が存在しません: {missing_columns}")
            sys.exit(1)

    return args


def train_model(args: Args, data: pl.DataFrame):
    """モデルの学習を実行"""
    # MLflowの設定
    if args.use_mlflow:
        print(f"\n🔧 MLflow設定:")
        print(f"  トラッキングURI: {args.mlflow_tracking_uri}")
        print(f"  実験名: {args.experiment_name}")
        
        # MLflowの設定
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        
        # 実行名の設定
        if not args.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"lightgbm_{args.table}_{timestamp}"
        
        print(f"  実行名: {args.run_name}")
        
        # MLflowの実行を開始
        mlflow.start_run(run_name=args.run_name)
        
        # パラメータを記録
        mlflow.log_param("table_name", args.table)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("feature_count", len(args.feature_columns))
        mlflow.log_param("data_size", len(data))
        mlflow.log_param("time_series_split", args.time_series_split)
        if args.time_series_split:
            mlflow.log_param("time_column", args.time_column)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("num_boost_round", 200)
        mlflow.log_param("early_stopping_rounds", 10)
        mlflow.log_param("test_size", 0.1)
        mlflow.log_param("random_state", 42)

    # ターゲットデータの準備
    target_data = data[args.target_column]
    target_manager = TargetManager(target_data=target_data)

    # 特徴量エンジニアリング
    print("\n🔧 特徴量エンジニアリング実行中...")
    engineered_data = feature_engineering(data, args.feature_columns, args.time_series_split)
    feature_manager = FeatureManager(initial_features=engineered_data)

    # 特徴量とターゲットの情報を表示
    print(f"\n{feature_manager}")
    print(f"\n{target_manager}")

    # LightGBMモデルを設定
    model = LightGBMRegressor(
        num_boost_round=200,
        early_stopping_rounds=10,
        params={"objective": "regression", "metric": "rmse", "verbose": 1, "seed": 42},
        verbose_eval=True,
    )

    # 自動的に学習情報をコメントとして追加
    print("\n📝 自動的に学習情報をコメントとして追加...")
    model.add_comment(f"データソース: {args.table} (SQLite)")
    model.add_comment(f"ターゲット列: {args.target_column}")
    model.add_comment(f"特徴量数: {len(args.feature_columns)}")
    model.add_comment(f"データサイズ: {len(data)} 行")
    model.add_comment(f"分割方法: {'時系列分割' if args.time_series_split else 'ランダム分割'}")
    if args.time_series_split:
        model.add_comment(f"時系列列: {args.time_column}")
    model.add_comment("特徴量エンジニアリング: 日時特徴量、時間特徴量、文字列変換、欠損値補完")
    model.add_comment("モデル: LightGBM (回帰)")
    model.add_comment("ハイパーパラメータ: デフォルト設定 (num_boost_round=200, early_stopping_rounds=10)")

    # カスタムコメントの入力
    print("\n📝 カスタムコメントの入力")
    print("=" * 40)
    print("学習時の情報として保存する追加コメントを入力してください。")
    print("（空行で入力を終了します）")

    skip_comments = input("追加コメントをスキップしますか？ (y/N): ").strip().lower()
    if skip_comments in ["y", "yes"]:
        print("⚠️  追加コメント入力をスキップしました")
        comments = []
    else:
        comments = []
        comment_count = 1
        while True:
            comment = input(f"追加コメント {comment_count}: ").strip()
            if not comment:
                break
            comments.append(comment)
            comment_count += 1

    # 追加コメントをモデルに追加
    for comment in comments:
        model.add_comment(comment)

    if comments:
        print(f"\n✅ {len(comments)}個の追加コメントを追加しました:")
        for i, comment in enumerate(comments, 1):
            print(f"  {i}. {comment}")
    else:
        print("\n⚠️  追加コメントは追加されませんでした")

    # 全コメントを表示
    print("\n📋 保存される全コメント:")
    all_comments = model.get_comments()
    for i, comment in enumerate(all_comments, 1):
        print(f"  {i}. {comment}")

    # パイプラインを構築
    pipeline = TreeModelPipeline(model=model, feature_manager=feature_manager, target_manager=target_manager)

    # モデル学習
    print("\n🔄 モデル学習中...")
    if args.time_series_split:
        print(f"  時系列分割を使用 (時系列列: {args.time_column})")
    else:
        print("  ランダム分割を使用")

    results = pipeline.train(
        test_size=0.1, random_state=42, time_series_split=args.time_series_split, time_column=args.time_column
    )

    # 結果表示
    print("\n📈 学習結果:")
    print(f"{results}")

    # MLflowにメトリクスを記録
    if args.use_mlflow:
        print("\n📊 MLflowにメトリクスを記録中...")
        # 結果からメトリクスを抽出
        if hasattr(results, 'train_rmse'):
            mlflow.log_metric("train_rmse", results.train_rmse)
        if hasattr(results, 'test_rmse'):
            mlflow.log_metric("test_rmse", results.test_rmse)
        if hasattr(results, 'train_r2'):
            mlflow.log_metric("train_r2", results.train_r2)
        if hasattr(results, 'test_r2'):
            mlflow.log_metric("test_r2", results.test_r2)
        if hasattr(results, 'train_mae'):
            mlflow.log_metric("train_mae", results.train_mae)
        if hasattr(results, 'test_mae'):
            mlflow.log_metric("test_mae", results.test_mae)
        
        # 特徴量重要度を記録
        importance = model.get_feature_importance()
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, imp) in enumerate(sorted_importance[:10]):
            mlflow.log_metric(f"feature_importance_{feature}", imp)
        
        # コメントをタグとして記録
        for i, comment in enumerate(all_comments):
            mlflow.set_tag(f"comment_{i+1}", comment)

    # 学習情報を表示
    print("\n📊 学習時の情報:")
    model.print_training_info()

    # 特徴量重要度を表示
    importance = model.get_feature_importance()
    print("\n🎯 特徴量重要度 (トップ10):")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importance[:10]:
        print(f"  {feature}: {imp:.4f}")

    return pipeline


def save_model(args: Args, pipeline):
    """モデルを保存"""
    if not args.no_save:
        try:
            save_path = Path(args.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # パイプライン全体を保存
            pipeline.save_model(save_path)
            print(f"\n💾 モデルを保存しました: {save_path}")

            # MLflowにモデルをアーティファクトとして保存
            if args.use_mlflow:
                print("\n📦 MLflowにモデルをアーティファクトとして保存中...")
                mlflow.log_artifact(str(save_path), "model")
                
                # 設定ファイルも保存
                config_path = save_path.with_suffix(".json")
                import json

                config = {
                    "table_name": args.table,
                    "target_column": args.target_column,
                    "feature_columns": args.feature_columns,
                    "model_type": "LightGBM",
                    "save_timestamp": str(datetime.now()),
                    "comments": pipeline.get_model().get_comments() if hasattr(pipeline.get_model(), "get_comments") else [],
                    "training_info": pipeline.get_model().get_training_info() if hasattr(pipeline.get_model(), "get_training_info") else {},
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                mlflow.log_artifact(str(config_path), "config")
                print(f"📋 設定情報をMLflowに保存しました: {config_path}")

            # 保存された学習情報を確認
            model = pipeline.get_model()
            if hasattr(model, "get_training_info"):
                training_info = model.get_training_info()
                print("📊 保存された学習情報:")
                print(f"  データサイズ: {training_info.get('data_size', 'N/A')}")
                print(f"  特徴量数: {training_info.get('feature_count', 'N/A')}")
                print(f"  学習日時: {training_info.get('training_timestamp', 'N/A')}")

            if hasattr(model, "get_comments"):
                comments = model.get_comments()
                if comments:
                    print(f"📝 保存されたコメント ({len(comments)}個):")
                    # 自動コメントと手動コメントを区別して表示
                    auto_comments = []
                    manual_comments = []

                    for comment in comments:
                        if any(
                            keyword in comment
                            for keyword in [
                                "データソース:",
                                "ターゲット列:",
                                "特徴量数:",
                                "データサイズ:",
                                "分割方法:",
                                "時系列列:",
                                "特徴量エンジニアリング:",
                                "モデル:",
                                "ハイパーパラメータ:",
                            ]
                        ):
                            auto_comments.append(comment)
                        else:
                            manual_comments.append(comment)

                    if auto_comments:
                        print("  🔧 自動生成コメント:")
                        for i, comment in enumerate(auto_comments, 1):
                            print(f"    {i}. {comment}")

                    if manual_comments:
                        print("  ✏️  手動入力コメント:")
                        for i, comment in enumerate(manual_comments, 1):
                            print(f"    {i}. {comment}")
                else:
                    print("📝 保存されたコメント: なし")

            # 設定情報も保存（MLflowを使用しない場合）
            if not args.use_mlflow:
                config_path = save_path.with_suffix(".json")
                import json

                config = {
                    "table_name": args.table,
                    "target_column": args.target_column,
                    "feature_columns": args.feature_columns,
                    "model_type": "LightGBM",
                    "save_timestamp": str(datetime.now()),
                    "comments": model.get_comments() if hasattr(model, "get_comments") else [],
                    "training_info": model.get_training_info() if hasattr(model, "get_training_info") else {},
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                print(f"📋 設定情報を保存しました: {config_path}")

        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
    else:
        print("\n⚠️  モデルは保存されませんでした (--no-save オプション)")
    
    # MLflowの実行を終了
    if args.use_mlflow:
        mlflow.end_run()
        print("\n✅ MLflowの実行を終了しました")


def main():
    # 引数の解析と検証
    args = parse_and_validate_args()

    # SQLiteからデータを読み込み
    try:
        print("\n📥 データ読み込み中...")
        data = load_data_from_sqlite_polars(args.db_path, args.table)
        print(f"✅ データ読み込み完了: {data.shape}")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        sys.exit(1)

    # データの基本情報表示
    print("\n📊 データの基本情報:")
    print(f"  形状: {data.shape}")
    print(f"  列名: {data.columns}")
    print(f"  ターゲット列: {args.target_column}")
    print(f"  特徴量列数: {len(args.feature_columns)}")

    # モデル学習
    pipeline = train_model(args, data)

    # モデル保存
    save_model(args, pipeline)

    # MLflowの使用状況を表示
    if args.use_mlflow:
        print(f"\n📊 MLflow情報:")
        print(f"  実験名: {args.experiment_name}")
        print(f"  実行名: {args.run_name}")
        print(f"  トラッキングURI: {args.mlflow_tracking_uri}")
        print("  MLflow UIで結果を確認できます:")
        print("    python -m mlflow ui")
        print("    ブラウザで http://localhost:5000 にアクセス")
    else:
        print(f"\n⚠️  MLflowは使用されませんでした (--no-mlflow オプションまたは --use-mlflow が指定されていません)")

    print("\n✅ 処理が完了しました!")


if __name__ == "__main__":
    main()
