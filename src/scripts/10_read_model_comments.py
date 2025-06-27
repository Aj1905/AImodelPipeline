#!/usr/bin/env python3
"""
MLflowから保存されたモデルのコメントと学習情報を読み取るスクリプト

このスクリプトは、MLflowのデータベースからモデルの学習情報とカスタムコメントを
表示します。
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))



def get_mlflow_runs(client, experiment_name=None):
    """MLflowから実験の実行履歴を取得"""
    try:
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attributes.start_time DESC"]
                )
            else:
                print(f"❌ 実験 '{experiment_name}' が見つかりません")
                return []
        else:
            # すべての実験から最新の実行を取得
            experiments = client.search_experiments()
            all_runs = []
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=5  # 各実験から最新5件
                )
                all_runs.extend(runs)
            runs = all_runs

        return runs
    except Exception as e:
        print(f"❌ MLflowから実行履歴の取得に失敗しました: {e}")
        return []


def load_and_display_model_info_from_mlflow(mlflow_uri, run_id=None, experiment_name=None):
    """MLflowからモデルを読み込んで情報を表示"""
    print(f"📥 MLflowからモデルを読み込み中: {mlflow_uri}")

    try:
        # MLflowクライアントを初期化
        client = MlflowClient(mlflow_uri)

        if run_id:
            # 特定のrun_idからモデルを読み込み
            print(f"🔍 実行ID: {run_id}")
            run = client.get_run(run_id)
            runs = [run]
        else:
            # 実験名または最新の実行から取得
            runs = get_mlflow_runs(client, experiment_name)
            if not runs:
                print("❌ 実行履歴が見つかりません")
                return False

        for run in runs:
            print(f"\n{'='*60}")
            print("📊 実行情報:")
            print(f"  実行ID: {run.info.run_id}")
            print(f"  実験名: {run.data.tags.get('mlflow.experiment.name', 'Unknown')}")
            print(f"  開始時刻: {run.info.start_time}")
            print(f"  ステータス: {run.info.status}")

            # パラメータを表示
            if run.data.params:
                print("\n🔧 パラメータ:")
                for key, value in run.data.params.items():
                    print(f"  {key}: {value}")

            # メトリクスを表示
            if run.data.metrics:
                print("\n📈 メトリクス:")
                for key, value in run.data.metrics.items():
                    print(f"  {key}: {value:.4f}")

            # タグを表示
            if run.data.tags:
                print("\n🏷️  タグ:")
                for key, value in run.data.tags.items():
                    if not key.startswith('mlflow.'):  # MLflow内部タグは除外
                        print(f"  {key}: {value}")

            # アーティファクトを表示
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                print("\n📦 アーティファクト:")
                for artifact in artifacts:
                    print(f"  - {artifact.path}")

                    # モデルファイルの場合、詳細情報を表示
                    if artifact.path.endswith('.pkl') or artifact.path.endswith('.json'):
                        try:
                            # モデルを読み込み
                            model_uri = f"runs:/{run.info.run_id}/{artifact.path}"
                            if artifact.path.endswith('.pkl'):
                                model = mlflow.sklearn.load_model(model_uri)
                            elif artifact.path.endswith('.json'):
                                model = mlflow.lightgbm.load_model(model_uri)

                            print("    ✅ モデル読み込み成功")

                            # モデルの属性を確認
                            if hasattr(model, 'feature_importances_'):
                                print(f"    📊 特徴量数: {len(model.feature_importances_)}")

                        except Exception as e:
                            print(f"    ⚠️  モデル読み込みエラー: {e}")

            print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"❌ MLflowからの読み込みエラー: {e}")
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="MLflowから保存されたモデルのコメントと学習情報を読み取る"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflowのデータベースURI (デフォルト: sqlite:///mlflow.db)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="特定の実行IDを指定"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="実験名を指定"
    )

    args = parser.parse_args()

    print("🔍 MLflowモデル情報読み取りツール")
    print("=" * 50)

    success = load_and_display_model_info_from_mlflow(
        args.mlflow_uri,
        args.run_id,
        args.experiment_name
    )

    if success:
        print("\n✨ 読み取り完了!")
    else:
        print("\n❌ 読み取りに失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
