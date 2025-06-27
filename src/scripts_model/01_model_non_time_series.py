#!/usr/bin/env python3
"""
非時系列データの機械学習パイプライン実行スクリプト

実行コマンド例:
# 基本的な実行(MLflow有効、チューニング有効)
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name mores_all \
    --target-column sales \
    --outerCV-splits 5

# チューニングをスキップする場合
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name mores_all \
    --target-column sales \
    --outerCV-splits 5 \
    --no-tuning

# MLflowを無効にする場合
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name mores_all \
    --target-column sales \
    --outerCV-splits 5 \
    --no-mlflow

# カスタム実験名を指定
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name mores_all \
    --target-column sales \
    --outerCV-splits 5 \
    --experiment-name "restaurant_sales_forecast"

# カスタム実行名を指定
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name mores_all \
    --target-column sales \
    --outerCV-splits 5 \
    --run-name "experiment_v1"

# MLflow UIで結果を確認
./start_mlflow_ui.sh
# ブラウザで http://localhost:5000 にアクセス
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.handlers.sqlite_handler import SQLiteHandler
from src.non_timeseries_pipeline.feature_manager import FeatureManager
from src.non_timeseries_pipeline.hyper_tuning import OptunaHyperTuner
from src.non_timeseries_pipeline.pipeline import TreeModelPipeline
from src.non_timeseries_pipeline.target_manager import TargetManager

# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "date_and_time",
    "date_year",
    "date_month",
    "date_day",
    "time",
    "dow",
    "holiday",
    "is_lunch",
    "is_dinner",
    "is_month_end",
    "is_weekend",
    "prev_year_same_weekday_sales_daily_sum",
]


# feature_engineering関数: テーブル変数を変換し、新たな特徴量を作成
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    必要な変換を行い、新しい特徴量を含むDataFrameを返します
    """
    df = df.copy()
    # 日付文字列 → datetime
    df["datetime"] = pd.to_datetime(df["date_and_time"], format="%Y-%m-%d %H:%M:%S")

    # ランチ時間フラグ(11-13時)
    df["is_lunch"] = df["time"].isin([11, 12, 13]).astype(int)

    # 曜日(0=月曜)
    df["weekday"] = df["datetime"].dt.weekday

    # 時間帯カテゴリ(0=<12時, 1=<15時, 2=<18時, 3=それ以外)
    df["time_category"] = df["time"].apply(lambda x: 0 if x < 12 else (1 if x < 15 else (2 if x < 18 else 3)))

    # 季節(0=冬,1=春,2=夏,3=秋)
    df["season"] = df["datetime"].dt.month.apply(
        lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3))
    )

    return df[["is_lunch", "weekday", "time_category", "season"]]


ENGINEERED_FEATURES = ["is_lunch"]
# ──────────────────────────────────────────────


def setup_mlflow(args) -> bool:
    """MLflowの設定を行う"""
    use_mlflow = args.use_mlflow or not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)

        if not args.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"non_timeseries_{args.table_name}_{timestamp}"

        mlflow.start_run(run_name=args.run_name)

        # 基本パラメータを記録
        mlflow.log_params(
            {
                "db_path": args.db_path,
                "table_name": args.table_name,
                "target_column": args.target_column,
                "outerCV_splits": args.outerCV_splits,
                "model_type": "LightGBM",
                "pipeline_type": "non_timeseries",
            }
        )
    return use_mlflow


def log_data_info(use_mlflow: bool, df: pd.DataFrame) -> None:
    """データ情報をMLflowに記録"""
    if use_mlflow:
        mlflow.log_params(
            {
                "data_rows": len(df),
                "data_columns": len(df.columns),
                "feature_columns": len(FEATURE_COLUMNS),
            }
        )


def log_tuning_results(
    use_mlflow: bool, best_params: dict, avg_score: float, std_score: float, all_scores: list
) -> None:
    """チューニング結果をMLflowに記録"""
    if use_mlflow:
        mlflow.log_metrics(
            {
                "tuning_avg_score": avg_score,
                "tuning_std_score": std_score,
                "tuning_best_score": max(all_scores),
                "tuning_worst_score": min(all_scores),
            }
        )

        # 最適パラメータを記録
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)


def log_final_metrics(use_mlflow: bool, train_metrics: dict, test_metrics: dict) -> None:
    """最終評価指標をMLflowに記録"""
    if use_mlflow:
        mlflow.log_metrics(
            {
                "train_r2": train_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "train_mse": train_metrics["mse"],
                "train_rmse": train_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_rmse": test_metrics["rmse"],
            }
        )


def log_feature_importance(use_mlflow: bool, feature_names: list, feature_importance: list) -> None:
    """特徴量重要度をMLflowに記録"""
    if use_mlflow:
        for feature_name, importance in zip(feature_names, feature_importance, strict=True):
            mlflow.log_metric(f"feature_importance_{feature_name}", importance)


def log_overfitting_analysis(use_mlflow: bool, overfitting: dict) -> None:
    """過学習分析をMLflowに記録"""
    if use_mlflow:
        mlflow.log_metrics(
            {
                "overfitting_r2_difference": overfitting["r2_difference"],
                "overfitting_rmse_difference": overfitting["rmse_difference"],
            }
        )
        mlflow.set_tag("is_overfitting", str(overfitting["is_overfitting"]))


def save_final_model(use_mlflow: bool, pipeline, experiment_name: str, run_name: str) -> None:
    """最終モデルをMLflowに保存"""
    if use_mlflow:
        # 最終モデルをアーティファクトとして保存
        model_path = "final_pipeline_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path, "pipeline_model")

        # MLflowの実行を終了
        mlflow.end_run()
        print(f"\n📊 MLflowに結果を記録しました - 実験名: {experiment_name}, 実行名: {run_name}")


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="非時系列データの機械学習パイプラインを実行します")
    parser.add_argument("--db-path", type=str, required=True, help="SQLiteデータベースファイルのパス")
    parser.add_argument("--table-name", type=str, required=True, help="読み込むテーブル名")
    parser.add_argument("--target-column", type=str, required=True, help="ターゲット列名")
    parser.add_argument("--outerCV-splits", type=int, default=5, help="外側のクロスバリデーション分割数")
    parser.add_argument("--use-mlflow", action="store_true", help="MLflowを使用する")
    parser.add_argument("--no-mlflow", action="store_true", help="MLflowを使用しない")
    parser.add_argument("--no-tuning", action="store_true", help="ハイパーパラメータチューニングをスキップする")
    parser.add_argument("--experiment-name", type=str, default="non_timeseries_pipeline", help="MLflowの実験名")
    parser.add_argument("--run-name", type=str, help="MLflowの実行名")
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflowのトラッキングURI"
    )
    args = parser.parse_args()

    # MLflowの設定
    use_mlflow = setup_mlflow(args)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ データベースファイルが見つかりません: {db_path}")
        sys.exit(1)

    with SQLiteHandler(db_path) as handler:
        # テーブルの存在確認
        if not handler.table_exists(args.table_name):
            print(f"❌ テーブル '{args.table_name}' が存在しません")
            # 利用可能なテーブル一覧を表示
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = handler.fetch_all(tables_query)
            if tables:
                print("利用可能なテーブル:")
                for table in tables:
                    print(f"  - {table[0]}")
            sys.exit(1)

        # テーブルの列情報を取得
        table_info = handler.get_table_info(args.table_name)
        columns = [col[1] for col in table_info]

        # ターゲット列の存在確認
        if args.target_column not in columns:
            print(f"❌ ターゲット列 '{args.target_column}' がテーブルに存在しません")
            print(f"利用可能な列: {', '.join(columns)}")
            sys.exit(1)

        # データを読み込み
        query = f"SELECT * FROM {args.table_name}"
        results = handler.fetch_all(query)
        df = pd.DataFrame(results, columns=columns)

    print(f"✅ データ読み込み完了: {len(df)} 行, {len(df.columns)} 列")
    print(f"テーブル: {args.table_name}")
    print(f"ターゲット列: {args.target_column}")

    # MLflowにデータ情報を記録
    log_data_info(use_mlflow, df)

    feature_manager = FeatureManager()
    target_manager = TargetManager()

    # 特徴量とターゲットを分離
    x_raw = df.drop(columns=[args.target_column])
    y_raw = df[args.target_column]

    # 前処理を実行
    x = feature_manager.transform(x_raw)
    y = target_manager.transform(y_raw)

    def estimator_factory() -> LGBMRegressor:
        """LightGBMモデルのファクトリ関数 - 警告を抑制する設定"""
        return LGBMRegressor(
            random_state=42,
            verbose=-1,  # 警告を抑制
            force_col_wise=True,  # 列方向の処理を強制
            min_child_samples=5,  # 最小サンプル数を設定
            min_split_gain=0.0,  # 分割ゲインの最小値を0に設定
        )

    # より適切なパラメータ範囲(Optuna用)
    param_ranges = {
        "n_estimators": (50, 200),  # 整数範囲
        "max_depth": (3, 10),  # 整数範囲
        "learning_rate": (0.01, 0.3),  # 浮動小数点範囲
        "min_child_samples": (5, 20),  # 整数範囲
        "subsample": (0.7, 1.0),  # 浮動小数点範囲
        "colsample_bytree": (0.7, 1.0),  # 浮動小数点範囲
        "reg_alpha": (0.0, 1.0),  # L1正則化
        "reg_lambda": (0.0, 1.0),  # L2正則化
    }

    if args.no_tuning:
        print("\n⏭️  ハイパーパラメータチューニングをスキップします")
        print("📋 デフォルトパラメータでモデルを作成します")

        # デフォルトパラメータでモデルを作成
        model = estimator_factory()

        # ネストCVで評価(チューニングなし)
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, x, y, cv=args.outerCV_splits, scoring="r2")

        avg_score = scores.mean()
        std_score = scores.std()
        all_scores = scores.tolist()

        print("\n📊 デフォルトパラメータでの評価結果:")
        print(f"平均スコア: {avg_score:.4f}")
        print(f"スコアの標準偏差: {std_score:.4f}")
        print(f"各フォールドのスコア: {[f'{score:.4f}' for score in all_scores]}")

        # 全データで学習
        model.fit(x, y)
        final_model = model

        # MLflowにチューニング結果を記録(デフォルトパラメータ)
        if use_mlflow:
            mlflow.log_params(
                {
                    "tuning_skipped": True,
                    "default_n_estimators": model.n_estimators,
                    "default_max_depth": model.max_depth,
                    "default_learning_rate": model.learning_rate,
                }
            )

    else:
        print("\n🔧 ネストCV + Optunaベイズ最適化でハイパーパラメータチューニング開始...")
        print("📋 手法: 外側CVでモデル評価 + 内側CVでベイズ最適化")
        print(f"外側CV分割数: {args.outerCV_splits}")
        print("内側CV分割数: 3")
        print("ベイズ最適化試行回数: 50")
        print("パラメータ範囲:")
        for param, range_val in param_ranges.items():
            print(f"  - {param}: {range_val}")

        tuner = OptunaHyperTuner(
            estimator_factory,
            param_ranges,
            outer_splits=args.outerCV_splits,
            inner_splits=3,
            n_trials=50,
            use_mlflow=use_mlflow,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            start_mlflow_run=False,  # 既存のMLflow実行を使用
        )
        tune_result = tuner.tune(x, y)

        best_params = tune_result["best_params"]
        avg_score = tune_result["avg_score"]
        std_score = tune_result["std_score"]
        all_scores = tune_result["all_scores"]
        final_model = tune_result["final_model"]

        print(f"\n🏆 最適パラメータ: {best_params}")
        print(f"📊 平均スコア: {avg_score:.4f}")
        print(f"📈 スコアの標準偏差: {std_score:.4f}")
        print(f"📈 各フォールドのスコア: {[f'{score:.4f}' for score in all_scores]}")

        # MLflowにチューニング結果を記録
        log_tuning_results(use_mlflow, best_params, avg_score, std_score, all_scores)

    # 最適パラメータでモデルを作成(final_modelは既に全データで学習済み)
    model = final_model

    pipeline = TreeModelPipeline(model=model, feature_manager=feature_manager, target_manager=target_manager)
    results = pipeline.train(test_size=0.1, random_state=42)

    # 結果を詳細に表示
    print("\n📊 学習データでの評価指標:")
    train_metrics = results["train_metrics"]
    print(f"   R²: {train_metrics['r2']:.4f}")
    print(f"   MAE: {train_metrics['mae']:.2f}")
    print(f"   MSE: {train_metrics['mse']:.2f}")
    print(f"   RMSE: {train_metrics['rmse']:.2f}")

    print("\n📊 テストデータでの評価指標:")
    test_metrics = results["test_metrics"]
    print(f"   R²: {test_metrics['r2']:.4f}")
    print(f"   MAE: {test_metrics['mae']:.2f}")
    print(f"   MSE: {test_metrics['mse']:.2f}")
    print(f"   RMSE: {test_metrics['rmse']:.2f}")

    # MLflowに最終評価指標を記録
    log_final_metrics(use_mlflow, train_metrics, test_metrics)

    # 特徴量重要度を表示
    print("\n🎯 特徴量重要度 (上位10件):")
    feature_importance = model.feature_importances_
    feature_names = x.columns.tolist()

    importance_pairs = list(zip(feature_names, feature_importance, strict=True))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_name, importance) in enumerate(importance_pairs[:10], 1):
        print(f"   {i:2d}. {feature_name:<30} {importance:.4f}")
        # MLflowに特徴量重要度を記録
        if use_mlflow:
            mlflow.log_metric(f"feature_importance_{feature_name}", importance)

    print("\n🔍 過学習の分析:")
    overfitting = results["overfitting_analysis"]
    if overfitting["is_overfitting"]:
        print("   ⚠️  過学習の可能性があります")
        for reason in overfitting["reasons"]:
            print(f"      - {reason}")
    else:
        print("   ✅ 過学習の兆候は見られません")

    print(f"   R²の差 (学習 - テスト): {overfitting['r2_difference']:.4f}")
    print(f"   RMSEの差 (テスト - 学習): {overfitting['rmse_difference']:.2f}")

    # MLflowに過学習分析を記録
    log_overfitting_analysis(use_mlflow, overfitting)

    # 最終モデルを保存
    save_final_model(use_mlflow, pipeline, args.experiment_name, args.run_name)


if __name__ == "__main__":
    main()
