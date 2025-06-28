"""
非時系列データの機械学習パイプライン実行スクリプト

実行例:
python src/scripts_model/01_model_non_time_series.py \
    --db-path data/database.sqlite \
    --table-name sales_data \
    --target-column sales \
    --outerCV-splits 5 \
    --use-mlflow \
    --experiment-name sales_prediction

オプション:
    --db-path: SQLiteデータベースファイルのパス
    --table-name: 読み込むテーブル名
    --target-column: ターゲット列名
    --outerCV-splits: 外側のCV分割数（デフォルト: 5）
    --use-mlflow: MLflowを使用する
    --no-mlflow: MLflowを使用しない
    --no-tuning: ハイパーパラメータチューニングをスキップする
    --experiment-name: MLflowの実験名（デフォルト: non_timeseries_pipeline）
    --run-name: MLflow実行名
    --mlflow-tracking-uri: MLflowトラッキングURI（デフォルト: sqlite:///mlflow.db）
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import joblib

import mlflow
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# 相対インポートを絶対インポートに変更
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.handlers.sqlite_handler import SQLiteHandler
from src.non_timeseries_pipeline.feature_manager import FeatureManager
from src.non_timeseries_pipeline.hyper_tuning import OptunaHyperTuner
from src.non_timeseries_pipeline.pipeline import TreeModelPipeline
from src.non_timeseries_pipeline.target_manager import TargetManager

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# 既存の列を特徴量として利用するリスト
FEATURE_COLUMNS = [
    "date_and_time",
    "dow_flag",
    "prev_year_same_weekday_sales_daily_sum",
    "time_flag",
    "Yokohama_Temperature",
    "is_weekend",
    "sales_trailing_ma_3",
    "sales_trailing_ma_14",
    "holiday",
]


# feature_engineering関数で作成する特徴量の名前
ENGINEERED_FEATURES = [
]

def feature_engineering(data: pl.DataFrame) -> pl.DataFrame:
    """一部の列を変換し、新たな特徴量列を返す"""
    df = data.clone()

    # 日付をdatetime型に変換
    df = df.with_columns(pl.col("date_and_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime"))

    # 新しい特徴量を作成
    engineered = df.with_columns(
        [
            # pl.when(pl.col("time").is_in([11, 12, 13])).then(1).otherwise(0).alias("is_lunch"),
            # pl.col("datetime").dt.weekday().alias("weekday"),
            # pl.when(pl.col("time") < 12)
            # .then(0)
            # .when(pl.col("time") < 15)
            # .then(1)
            # .when(pl.col("time") < 18)
            # .then(2)
            # .otherwise(3)
            # .alias("time_category"),
            # pl.when(pl.col("date_month").is_in([12, 1, 2]))
            # .then(0)
            # .when(pl.col("date_month").is_in([3, 4, 5]))
            # .then(1)
            # .when(pl.col("date_month").is_in([6, 7, 8]))
            # .then(2)
            # .otherwise(3)
            # .alias("season"),
        ]
    )

    return engineered.select(ENGINEERED_FEATURES)

def setup_mlflow(args) -> bool:
    """MLflow の設定を行い、必要なら run を開始する"""
    use_mlflow = args.use_mlflow or not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        if not args.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"non_timeseries_{args.table_name}_{timestamp}"
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_params({
            "db_path": args.db_path,
            "table_name": args.table_name,
            "target_column": args.target_column,
            "outerCV_splits": args.outerCV_splits,
            "model_type": "LightGBM",
            "pipeline_type": "non_timeseries",
        })
    return use_mlflow

def log_data_info(use_mlflow: bool, df: pd.DataFrame, feature_columns: int) -> None:
    """データ情報をMLflowに記録"""
    if use_mlflow:
        mlflow.log_params({
            "data_rows": len(df),
            "data_columns": len(df.columns),
            "feature_columns": feature_columns,
        })

def log_tuning_results(use_mlflow: bool, best_params: dict, avg_score: float, std_score: float, all_scores: list) -> None:
    """チューニング結果をMLflowに記録"""
    if use_mlflow:
        mlflow.log_metrics({
            "tuning_avg_score": avg_score,
            "tuning_std_score": std_score,
            "tuning_best_score": max(all_scores),
            "tuning_worst_score": min(all_scores),
        })
        for name, val in best_params.items():
            mlflow.log_param(f"best_{name}", val)

def log_final_metrics(use_mlflow: bool, train_metrics: dict, test_metrics: dict) -> None:
    """最終評価指標をMLflowに記録"""
    if use_mlflow:
        mlflow.log_metrics({
            "train_r2":    train_metrics["r2"],
            "train_mae":   train_metrics["mae"],
            "train_mse":   train_metrics["mse"],
            "train_rmse":  train_metrics["rmse"],
            "test_r2":     test_metrics["r2"],
            "test_mae":    test_metrics["mae"],
            "test_mse":    test_metrics["mse"],
            "test_rmse":   test_metrics["rmse"],
        })

def log_feature_importance(use_mlflow: bool, feature_names: list, importances: list) -> None:
    """特徴量重要度をMLflowに記録"""
    if use_mlflow:
        for name, imp in zip(feature_names, importances, strict=True):
            mlflow.log_metric(f"feature_importance_{name}", imp)

def main():
    parser = argparse.ArgumentParser(description="非時系列データの機械学習パイプラインを実行し、モデルを保存します")
    parser.add_argument("--db-path",      type=str, required=True,  help="SQLiteデータベースファイルのパス")
    parser.add_argument("--table-name",   type=str, required=True,  help="読み込むテーブル名")
    parser.add_argument("--target-column",type=str, required=True,  help="ターゲット列名")
    parser.add_argument("--outerCV-splits",type=int, default=5,    help="外側のCV分割数")
    parser.add_argument("--use-mlflow",   action="store_true",     help="MLflowを使用する")
    parser.add_argument("--no-mlflow",    action="store_true",     help="MLflowを使用しない")
    parser.add_argument("--no-tuning",    action="store_true",     help="ハイパーパラメータチューニングをスキップする")
    parser.add_argument("--experiment-name", type=str, default="non_timeseries_pipeline", help="MLflowの実験名")
    parser.add_argument("--run-name",        type=str, help="MLflow実行名")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflowトラッキングURI")
    args = parser.parse_args()

    # MLflow設定
    use_mlflow = setup_mlflow(args)

    # データベース存在チェック
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ データベースが見つかりません: {db_path}")
        sys.exit(1)

    # データ読み込み
    with SQLiteHandler(db_path) as handler:
        if not handler.table_exists(args.table_name):
            print(f"❌ テーブル '{args.table_name}' が存在しません")
            sys.exit(1)
        cols = [col[1] for col in handler.get_table_info(args.table_name)]
        rows = handler.fetch_all(f"SELECT * FROM {args.table_name}")
    df = pd.DataFrame(rows, columns=cols)
    print(f"✅ データ読込完了: {len(df)} 行, {len(df.columns)} 列")

    # 指定された特徴量のみを選択
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    
    if missing_features:
        print(f"⚠️ 指定された特徴量が見つかりません: {missing_features}")
    
    print(f"📊 使用する特徴量: {available_features}")
    
    # 指定された特徴量とターゲット列のみを選択
    selected_columns = available_features + [args.target_column]
    df_selected = df[selected_columns].copy()
    
    print("🔧 特徴量エンジニアリング実行中...")
    df_pl = pl.from_pandas(df_selected)
    engineered_features = feature_engineering(df_pl)
    engineered_features_pd = engineered_features.to_pandas()
    
    X_raw = pd.concat([df_selected[available_features], engineered_features_pd], axis=1)
    y_raw = df_selected[args.target_column]
    
    print(f"✅ 特徴量エンジニアリング完了: {len(X_raw.columns)} 列")
    print(f"   元の特徴量: {len(available_features)} 列")
    print(f"   エンジニアリング特徴量: {len(ENGINEERED_FEATURES)} 列")

    # 前処理
    fm = FeatureManager()
    tm = TargetManager()
    X_all = fm.transform(X_raw)
    y_all = tm.transform(y_raw)

    # ── ① ホールドアウト用テストセットを切り出す ──
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    log_data_info(use_mlflow, X_dev, feature_columns=len(X_dev.columns))

    # モデル定義
    def estimator_factory() -> LGBMRegressor:
        return LGBMRegressor(
            random_state=42,
            verbose=-1,
            force_col_wise=True,
            min_child_samples=5,
            min_split_gain=0.0,
        )

    default_param_ranges = {
        "n_estimators":       (50, 200),
        "max_depth":          (3,  10),
        "learning_rate":      (0.01, 0.3),
        "min_child_samples":  (5, 20),
        "subsample":          (0.7, 1.0),
        "colsample_bytree":   (0.7, 1.0),
        "reg_alpha":          (0.0, 1.0),
        "reg_lambda":         (0.0, 1.0),
    }

    # チューニング or デフォルト学習
    if args.no_tuning:
        print("⏭️ ハイパーパラメータチューニングをスキップします")
        model = estimator_factory()
        model.fit(X_dev, y_dev)
        best_params = model.get_params()
        if use_mlflow:
            mlflow.log_param("tuning_skipped", True)
    else:
        print("🔧 ネストCV + Optunaでチューニング開始...")
        tuner = OptunaHyperTuner(
            estimator_factory,
            default_param_ranges,
            outer_splits=args.outerCV_splits,
            inner_splits=3,
            n_trials=50,
            use_mlflow=use_mlflow,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            start_mlflow_run=False,
        )
        tune_res = tuner.tune(X_dev, y_dev)
        model       = tune_res["final_model"]
        best_params = tune_res["best_params"]
        avg_score   = tune_res["avg_score"]
        std_score   = tune_res["std_score"]
        all_scores  = tune_res["all_scores"]
        log_tuning_results(use_mlflow, best_params, avg_score, std_score, all_scores)

    # ── ② 開発用データ全体で再学習 ──
    model.fit(X_dev, y_dev)

    # ── ③ ホールドアウトテストで評価 ──
    # Dev
    y_dev_pred = model.predict(X_dev)
    train_m = {
        "r2":   r2_score(y_dev, y_dev_pred),
        "mae":  mean_absolute_error(y_dev, y_dev_pred),
        "mse":  mean_squared_error(y_dev, y_dev_pred),
        "rmse": mean_squared_error(y_dev, y_dev_pred) ** 0.5,
    }
    # Test
    y_test_pred = model.predict(X_test)
    test_m = {
        "r2":   r2_score(y_test, y_test_pred),
        "mae":  mean_absolute_error(y_test, y_test_pred),
        "mse":  mean_squared_error(y_test, y_test_pred),
        "rmse": mean_squared_error(y_test, y_test_pred) ** 0.5,
    }

    # コンソール出力
    print("\n📊 学習データ評価:")
    print(f"   R²:   {train_m['r2']:.4f}, MAE: {train_m['mae']:.2f}, MSE: {train_m['mse']:.2f}, RMSE: {train_m['rmse']:.2f}")
    print("\n📊 テストデータ評価:")
    print(f"   R²:   {test_m['r2']:.4f}, MAE: {test_m['mae']:.2f}, MSE: {test_m['mse']:.2f}, RMSE: {test_m['rmse']:.2f}")

    log_final_metrics(use_mlflow, train_m, test_m)

    # 特徴量重要度
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        names = X_dev.columns.tolist()
        pairs = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)[:10]
        print("\n🎯 特徴量重要度 (上位10件):")
        for i, (n, v) in enumerate(pairs, 1):
            print(f"   {i:2d}. {n:<30} {v:.4f}")
        log_feature_importance(use_mlflow, names, fi)

    # アーティファクト保存
    joblib.dump(fm, ARTIFACT_DIR/"feature_manager.pkl")
    joblib.dump(tm, ARTIFACT_DIR/"target_manager.pkl")
    joblib.dump(model, ARTIFACT_DIR/"model.pkl")

    metadata = {
        "feature_columns":     X_dev.columns.tolist(),
        "original_features":   available_features,
        "engineered_features": ENGINEERED_FEATURES,
        "categorical_columns": fm.categorical_columns,
        "date_columns":        fm.date_columns,
        "best_params":         best_params,
        "run_datetime":        datetime.now().isoformat()
    }
    with open(ARTIFACT_DIR/"metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # MLflow run 終了
    if use_mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    main()
