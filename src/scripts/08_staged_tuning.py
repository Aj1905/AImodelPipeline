#!/usr/bin/env python3
"""
段階的チューニングスクリプト

このスクリプトは、機械学習モデルの段階的チューニングを実行します。

実行コマンド例:
    python src/scripts/08_staged_tuning.py --action new --table TABLE_NAME --target TARGET_COLUMN --project PROJECT_NAME
    python src/scripts/08_staged_tuning.py --action list
    python src/scripts/08_staged_tuning.py --action details --project PROJECT_NAME
    python src/scripts/08_staged_tuning.py --action compare
    python src/scripts/08_staged_tuning.py --action importance --project PROJECT_NAME
    python src/scripts/08_staged_tuning.py --action new --table my_table --target target_col --db-file data/database.sqlite
    python src/scripts/08_staged_tuning.py --action new --table sales_data --target sales --project sales_prediction
    python src/scripts/08_staged_tuning.py --action details --project weather_forecast
    python src/scripts/08_staged_tuning.py --action importance --project customer_analysis

使用方法:
    python 06_staged_tuning.py --action new --table TABLE_NAME --target TARGET_COLUMN
        --project PROJECT_NAME
    python 06_staged_tuning.py --action list
    python 06_staged_tuning.py --action details --project PROJECT_NAME
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.utils.data_loader import (
    load_data_from_table,
    validate_db_path,
)
from src.data.utils.interactive_selector import (
    _get_user_choice,
)

# ============================================================================
# 定数定義
# ============================================================================

DEFAULT_DB_PATH = "data/database.sqlite"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# ============================================================================
# 関数定義
# ============================================================================


def simple_ml_pipeline(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> dict[str, Any]:
    """
    シンプルな機械学習パイプライン

    Args:
        x: 特徴量データ
        y: ターゲットデータ
        test_size: テストデータの割合

    Returns:
        学習結果の辞書
    """
    # データ分割
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # モデル学習
    model = LGBMRegressor(random_state=42, verbose=-1)
    model.fit(x_train, y_train)

    # 予測
    y_pred = model.predict(x_test)

    # 評価指標計算
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred
    }


def staged_tuning(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str]
) -> dict[str, Any]:
    """
    段階的チューニングを実行

    Args:
        df: データフレーム
        target_column: ターゲット列名
        feature_columns: 特徴量列名のリスト

    Returns:
        チューニング結果の辞書
    """
    # 基本モデルの学習
    x = df[feature_columns]
    y = df[target_column]
    basic_results = simple_ml_pipeline(x, y)

    # 特徴量重要度による特徴量選択
    feature_importance = basic_results["model"].feature_importances_
    sorted_features = sorted(
        zip(feature_columns, feature_importance, strict=False),
        key=lambda x: x[1],
        reverse=True
    )

    # 上位50%の特徴量を選択
    top_n = max(1, len(feature_columns) // 2)
    top_features = [feature for feature, _ in sorted_features[:top_n]]

    # 選択された特徴量でモデル再学習
    x_selected = df[top_features]
    selected_results = simple_ml_pipeline(x_selected, y)

    # 結果を辞書にまとめる
    feature_importance_dict = {
        feature: float(importance)
        for feature, importance in sorted_features
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "feature_importance": feature_importance_dict,
        "selected_features": top_features,
        "basic_model": {
            "mse": basic_results["mse"],
            "r2": basic_results["r2"]
        },
        "selected_model": {
            "mse": selected_results["mse"],
            "r2": selected_results["r2"]
        },
        "total_features": len(feature_columns),
        "selected_feature_count": len(top_features)
    }


def list_projects():
    """プロジェクト一覧を表示"""
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiments = client.list_experiments()
        if not experiments:
            print("プロジェクトが見つかりません")
            return

        print("\n📋 プロジェクト一覧:")
        print("=" * 60)
        for exp in experiments:
            print(f"📁 {exp.name}")
            print(f"   実験ID: {exp.experiment_id}")
            print(f"   作成日時: {exp.creation_time}")
            print()

    except Exception as e:
        print(f"プロジェクト一覧の取得に失敗しました: {e}")


def show_project_details(project_name: str):
    """プロジェクトの詳細を表示"""
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # 実験を取得
        experiment = client.get_experiment_by_name(project_name)
        if not experiment:
            print(f"プロジェクト '{project_name}' が見つかりません")
            return

        # 実行履歴を取得
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"]
        )

        if not runs:
            print(f"プロジェクト '{project_name}' に実行履歴がありません")
            return

        print(f"\n📊 プロジェクト詳細: {project_name}")
        print("=" * 60)

        for run in runs:
            try:
                print(f"🔄 実行ID: {run.info.run_id}")
                print(f"   実行日時: {run.info.start_time}")
                print(
                    f"   基本モデル R²: {run.data.metrics.get('basic_r2', 'N/A')}"
                )
                print(f"   選択モデル R²: {run.data.metrics.get('selected_r2', 'N/A')}")
                print(f"   特徴量数: {run.data.params.get('total_features', 'N/A')}")
                print(f"   選択特徴量数: {run.data.params.get('selected_features', 'N/A')}")
                print()
            except Exception as e:
                print(f"   実行情報の取得に失敗: {e}")

    except Exception as e:
        print(f"プロジェクト詳細の取得に失敗しました: {e}")


def compare_projects():
    """プロジェクト間の比較"""
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiments = client.list_experiments()
        if not experiments:
            print("比較対象のプロジェクトが見つかりません")
            return

        project_names = [p.name for p in experiments]
        choice = _get_user_choice(
            "詳細を表示するプロジェクトを選択してください:",
            project_names
        )

        if choice > 0:
            selected_project = project_names[choice - 1]
            show_project_details(selected_project)

    except Exception as e:
        print(f"プロジェクト比較に失敗しました: {e}")


def show_feature_importance(project_name: str):
    """特徴量重要度を表示"""
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(project_name)
        if not experiment:
            print(f"プロジェクト '{project_name}' が見つかりません")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            print(f"プロジェクト '{project_name}' に実行履歴がありません")
            return

        run = runs[0]
        results = json.loads(run.data.params.get("results", "{}"))

        if "feature_importance" in results:
            feature_importance = results.get("feature_importance", {})
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n🎯 特徴量重要度 (プロジェクト: {project_name})")
            print("=" * 60)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f"特徴量重要度の表示に失敗しました: {e}")


def show_feature_importance_comparison():
    """プロジェクト間の特徴量重要度比較"""
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiments = client.list_experiments()
        if not experiments:
            print("比較対象のプロジェクトが見つかりません")
            return

        project_names = [p.name for p in experiments]
        choice = _get_user_choice(
            "特徴量重要度を表示するプロジェクトを選択してください:",
            project_names
        )

        if choice > 0:
            selected_project = project_names[choice - 1]
            show_feature_importance(selected_project)

    except Exception as e:
        print(f"特徴量重要度比較に失敗しました: {e}")


def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="段階的チューニングスクリプト")
    parser.add_argument(
        "--action",
        choices=["new", "list", "details", "compare", "importance"],
        required=True,
        help="実行するアクション"
    )
    parser.add_argument(
        "--table",
        help="対象テーブル名 (action=newの場合に使用)"
    )
    parser.add_argument(
        "--target",
        help="ターゲット列名 (action=newの場合に使用)"
    )
    parser.add_argument(
        "--project",
        help="プロジェクト名 (action=newまたはdetailsの場合に使用)"
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLiteデータベースファイルパス"
    )
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_arguments()

    if args.action == "list":
        list_projects()
        return

    if args.action == "details":
        if not args.project:
            print("--project オプションが必要です")
            return
        show_project_details(args.project)
        return

    if args.action == "compare":
        compare_projects()
        return

    if args.action == "importance":
        if not args.project:
            show_feature_importance_comparison()
        else:
            show_feature_importance(args.project)
        return

    if args.action == "new":
        if not args.table or not args.target:
            print("--table と --target オプションが必要です")
            return
        execute_new_staged_tuning(
            args.table,
            args.target,
            args.project,
            args.db_file
        )


def execute_new_staged_tuning(
    table_name: str,
    target_column: str,
    project_name: str | None = None,
    db_path: str | None = None
):
    """新しい段階的チューニングを実行"""
    # データベースパスの設定
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = Path(db_path).expanduser().resolve()
    if not validate_db_path(db_path):
        return

    # データ読み込み
    print(f"\n📊 データ読み込み中: {table_name}")
    df = load_data_from_table(db_path, table_name)
    if df is None or df.empty:
        print("データの読み込みに失敗しました")
        return

    print(f"読み込み完了: {df.shape[0]}行 x {df.shape[1]}列")

    # ターゲット列の存在確認
    if target_column not in df.columns:
        print(f"ターゲット列 '{target_column}' が見つかりません")
        return

    # 特徴量列の取得(目的変数を除く)
    feature_columns = [col for col in df.columns if col != target_column]

    if not feature_columns:
        print("特徴量列が見つかりません")
        return

    print(f"特徴量数: {len(feature_columns)}")

    # プロジェクト名の決定
    if project_name is None:
        project_name = f"{table_name}_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 段階チューニング実行
    print(f"\n🔄 段階的チューニング実行中: {project_name}")
    results = staged_tuning(df, target_column, feature_columns)

    # MLflowに記録
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        mlflow.set_experiment(project_name)

        with mlflow.start_run():
            # パラメータを記録
            mlflow.log_params({
                "table_name": table_name,
                "target_column": target_column,
                "total_features": results["total_features"],
                "selected_features": results["selected_feature_count"],
                "results": json.dumps(results)
            })

            # メトリクスを記録
            mlflow.log_metrics({
                "basic_mse": results["basic_model"]["mse"],
                "basic_r2": results["basic_model"]["r2"],
                "selected_mse": results["selected_model"]["mse"],
                "selected_r2": results["selected_model"]["r2"]
            })

        print("✅ MLflowに記録完了")

    except Exception as e:
        print(f"MLflowへの記録に失敗しました: {e}")

    # 結果表示
    print("\n📊 チューニング結果:")
    print(f"  基本モデル R²: {results['basic_model']['r2']:.4f}")
    print(f"  選択モデル R²: {results['selected_model']['r2']:.4f}")
    print(f"  特徴量数: {results['total_features']} → {results['selected_feature_count']}")

    # 特徴量重要度トップ5を表示
    feature_importance = results.get("feature_importance", {})
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n🎯 特徴量重要度 (トップ5):")
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
