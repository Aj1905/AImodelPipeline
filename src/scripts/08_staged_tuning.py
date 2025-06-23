#!/usr/bin/env python3
"""
段階チューニング機能を使用するスクリプト

このスクリプトは、db_utils.pyのみを使用して段階的なハイパーパラメータチューニングを実行し、
結果をtuning_resultディレクトリに保存します。

使用方法:
    python 06_staged_tuning.py --action new --table TABLE_NAME --target TARGET_COLUMN --project PROJECT_NAME
    python 06_staged_tuning.py --action list
    python 06_staged_tuning.py --action details --project PROJECT_NAME
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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


def simple_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """シンプルなデータ前処理"""
    print("データ前処理を実行中...")

    # 数値列のみを選択
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        print("❌ 数値列が不足しています")
        return pd.DataFrame()

    # 欠損値を処理
    df_processed = df[numeric_columns].copy()
    df_processed = df_processed.fillna(df_processed.mean())

    print(f"✓ 前処理完了 (使用列数: {len(numeric_columns)})")
    return df_processed


def simple_ml_pipeline(x: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict[str, Any]:
    """シンプルな機械学習パイプライン"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # データ分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # モデル学習
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # 予測
    y_pred = model.predict(x_test)

    # 評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"mse": mse, "r2": r2, "model": model}


def staged_tuning(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> dict[str, Any]:
    """段階的なチューニングを実行"""
    print("\n=== 段階チューニング開始 ===")
    print(f"目的変数: {target_column}")
    print(f"特徴量数: {len(feature_columns)}")

    # データ準備
    x = df[feature_columns]
    y = df[target_column]

    # 段階1: 基本モデル
    print("\n段階1: 基本モデル")
    basic_results = simple_ml_pipeline(x, y)
    print(f"  MSE: {basic_results['mse']:.4f}")
    print(f"  R²: {basic_results['r2']:.4f}")

    # 段階2: 特徴量選択
    print("\n段階2: 特徴量選択")
    feature_importance = basic_results["model"].feature_importances_
    sorted_features = sorted(zip(feature_columns, feature_importance, strict=False), key=lambda x: x[1], reverse=True)

    # 上位50%の特徴量を選択
    top_features = [f[0] for f in sorted_features[: len(sorted_features) // 2]]
    print(f"  選択された特徴量数: {len(top_features)}")

    if len(top_features) > 0:
        x_selected = x[top_features]
        selected_results = simple_ml_pipeline(x_selected, y)
        print(f"  MSE: {selected_results['mse']:.4f}")
        print(f"  R²: {selected_results['r2']:.4f}")
    else:
        selected_results = basic_results
        top_features = feature_columns

    # 結果をまとめる
    results = {
        "timestamp": datetime.now().isoformat(),
        "target_column": target_column,
        "initial_features": feature_columns,
        "selected_features": top_features,
        "basic_model": {"mse": basic_results["mse"], "r2": basic_results["r2"]},
        "selected_model": {"mse": selected_results["mse"], "r2": selected_results["r2"]},
        "feature_importance": dict(sorted_features),
    }

    return results


def save_tuning_results(results: dict[str, Any], project_name: str):
    """チューニング結果を保存"""
    tuning_dir = project_root / "tuning_result"
    tuning_dir.mkdir(exist_ok=True)

    # プロジェクトディレクトリを作成
    project_dir = tuning_dir / project_name
    project_dir.mkdir(exist_ok=True)

    # 結果をJSONファイルに保存
    results_file = project_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 結果を保存しました: {results_file}")


def list_tuning_projects():
    """チューニングプロジェクトの一覧を表示"""
    tuning_dir = project_root / "tuning_result"

    if not tuning_dir.exists():
        print("チューニング結果ディレクトリが存在しません")
        return

    projects = [d for d in tuning_dir.iterdir() if d.is_dir()]

    if not projects:
        print("チューニングプロジェクトが見つかりません")
        return

    print("\n=== チューニングプロジェクト一覧 ===")
    for i, project in enumerate(projects, 1):
        results_file = project / "results.json"
        if results_file.exists():
            try:
                with open(results_file, encoding="utf-8") as f:
                    results = json.load(f)
                print(f"{i}. {project.name}")
                print(f"   目的変数: {results.get('target_column', 'N/A')}")
                print(f"   実行日時: {results.get('timestamp', 'N/A')}")
                print(f"   基本モデル R²: {results.get('basic_model', {}).get('r2', 'N/A')}")
            except Exception as e:
                print(f"{i}. {project.name} (読み込みエラー: {e})")
        else:
            print(f"{i}. {project.name} (結果ファイルなし)")


def display_project_details():
    """プロジェクト詳細を表示"""
    tuning_dir = project_root / "tuning_result"

    if not tuning_dir.exists():
        print("チューニング結果ディレクトリが存在しません")
        return

    projects = [d for d in tuning_dir.iterdir() if d.is_dir()]

    if not projects:
        print("チューニングプロジェクトが見つかりません")
        return

    project_names = [p.name for p in projects]
    choice = _get_user_choice("詳細を表示するプロジェクトを選択してください:", project_names)
    selected_project = projects[choice - 1]

    results_file = selected_project / "results.json"
    if not results_file.exists():
        print("結果ファイルが見つかりません")
        return

    try:
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)

        print(f"\n=== プロジェクト詳細: {selected_project.name} ===")
        print(f"実行日時: {results.get('timestamp', 'N/A')}")
        print(f"目的変数: {results.get('target_column', 'N/A')}")
        print(f"初期特徴量数: {len(results.get('initial_features', []))}")
        print(f"選択特徴量数: {len(results.get('selected_features', []))}")

        print("\n基本モデル:")
        basic_model = results.get("basic_model", {})
        print(f"  MSE: {basic_model.get('mse', 'N/A')}")
        print(f"  R²: {basic_model.get('r2', 'N/A')}")

        print("\n選択モデル:")
        selected_model = results.get("selected_model", {})
        print(f"  MSE: {selected_model.get('mse', 'N/A')}")
        print(f"  R²: {selected_model.get('r2', 'N/A')}")

        print("\n特徴量重要度 (上位10件):")
        feature_importance = results.get("feature_importance", {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f"結果ファイルの読み込みエラー: {e}")


def display_project_details_by_name(project_name: str):
    """プロジェクト名を指定して詳細を表示"""
    tuning_dir = project_root / "tuning_result"
    selected_project = tuning_dir / project_name

    if not tuning_dir.exists():
        print("チューニング結果ディレクトリが存在しません")
        return

    if not selected_project.exists():
        print(f"プロジェクト '{project_name}' が見つかりません")
        return

    results_file = selected_project / "results.json"
    if not results_file.exists():
        print("結果ファイルが見つかりません")
        return

    try:
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)

        print(f"\n=== プロジェクト詳細: {selected_project.name} ===")
        print(f"実行日時: {results.get('timestamp', 'N/A')}")
        print(f"目的変数: {results.get('target_column', 'N/A')}")
        print(f"初期特徴量数: {len(results.get('initial_features', []))}")
        print(f"選択特徴量数: {len(results.get('selected_features', []))}")

        print("\n基本モデル:")
        basic_model = results.get("basic_model", {})
        print(f"  MSE: {basic_model.get('mse', 'N/A')}")
        print(f"  R²: {basic_model.get('r2', 'N/A')}")

        print("\n選択モデル:")
        selected_model = results.get("selected_model", {})
        print(f"  MSE: {selected_model.get('mse', 'N/A')}")
        print(f"  R²: {selected_model.get('r2', 'N/A')}")

        print("\n特徴量重要度 (上位10件):")
        feature_importance = results.get("feature_importance", {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f"結果ファイルの読み込みエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="段階チューニングスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s --action new --table sales_data --target revenue --project my_tuning
  %(prog)s --action list
  %(prog)s --action details --project my_tuning
        """,
    )

    parser.add_argument(
        "--action",
        choices=["new", "list", "details"],
        required=True,
        help="実行する操作: new (新しいチューニング), list (プロジェクト一覧), details (詳細表示)",
    )

    parser.add_argument("--table", help="使用するテーブル名 (action=newの場合に必要)")

    parser.add_argument("--target", help="目的変数名 (action=newの場合に必要)")

    parser.add_argument("--project", help="プロジェクト名 (action=newまたはdetailsの場合に使用)")

    parser.add_argument(
        "--db-path",
        default=str(project_root / "data" / "database.sqlite"),
        help="データベースファイルのパス (デフォルト: data/database.sqlite)",
    )

    args = parser.parse_args()

    if args.action == "new":
        if not args.table or not args.target:
            print("❌ 新しいチューニングには --table と --target が必要です")
            sys.exit(1)
        execute_new_staged_tuning(args.table, args.target, args.project, args.db_path)
    elif args.action == "list":
        list_tuning_projects()
    elif args.action == "details":
        if not args.project:
            print("❌ 詳細表示には --project が必要です")
            sys.exit(1)
        display_project_details_by_name(args.project)
    else:
        print("❌ 無効な操作です")
        sys.exit(1)


def execute_new_staged_tuning(table_name: str, target_column: str, project_name: str = None, db_path: str = None):
    """新しい段階チューニングを実行"""
    print("\n=== 新しい段階チューニング ===")

    if db_path is None:
        db_path = str(project_root / "data" / "database.sqlite")

    if not validate_db_path(db_path):
        return

    # データ読み込み
    print(f"\nテーブル '{table_name}' からデータを読み込み中...")
    data = load_data_from_table(db_path, table_name)

    if data.empty:
        print("❌ データの読み込みに失敗しました")
        return

    # データ前処理
    data_processed = simple_data_preprocessing(data)

    if data_processed.empty:
        print("❌ データ前処理に失敗しました")
        return

    # 目的変数の存在確認
    if target_column not in data_processed.columns:
        print(f"❌ 目的変数 '{target_column}' がデータに存在しません")
        print(f"利用可能な列: {', '.join(data_processed.columns)}")
        return

    # 特徴量の選択
    feature_columns = [col for col in data_processed.columns if col != target_column]

    if len(feature_columns) == 0:
        print("❌ 特徴量が不足しています")
        return

    print(f"\n目的変数: {target_column}")
    print(f"特徴量数: {len(feature_columns)}")

    # 段階チューニング実行
    print("\n段階チューニングを開始します...")
    try:
        results = staged_tuning(data_processed, target_column, feature_columns)

        # プロジェクト名の設定
        if not project_name:
            project_name = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 結果を保存
        save_tuning_results(results, project_name)

        print("\n✓ 段階チューニングが完了しました")
        print(f"最良スコア (R²): {results['selected_model']['r2']:.4f}")

    except Exception as e:
        print(f"\n❌ 段階チューニングエラー: {e}")


if __name__ == "__main__":
    main()
