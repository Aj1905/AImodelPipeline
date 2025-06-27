from collections.abc import Callable
from typing import Any
from datetime import datetime

import pandas as pd
import optuna
import mlflow
from sklearn.model_selection import cross_val_score


class OptunaHyperTuner:
    """Hyper parameter tuning using Optuna with nested cross validation and MLflow integration."""

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        param_ranges: dict[str, Any],
        outer_splits: int = 5,
        inner_splits: int = 3,
        n_trials: int = 50,
        scoring: str = "r2",
        random_state: int = 42,
        use_mlflow: bool = True,
        experiment_name: str = "hyperparameter_tuning",
        run_name: str = None,
        start_mlflow_run: bool = False,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.param_ranges = param_ranges
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_trials = n_trials
        self.scoring = scoring
        self.random_state = random_state
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.start_mlflow_run = start_mlflow_run

    def _setup_mlflow(self) -> None:
        """MLflowの設定を行う"""
        if not self.use_mlflow:
            return

        # MLflowの設定
        mlflow.set_experiment(self.experiment_name)
        
        # 実行名の設定
        if not self.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"optuna_tuning_{timestamp}"

        # start_mlflow_runがTrueの場合のみ新しい実行を開始
        if self.start_mlflow_run:
            try:
                current_run = mlflow.active_run()
                if current_run is None:
                    # アクティブな実行がない場合のみ新しい実行を開始
                    mlflow.start_run(run_name=self.run_name)
                    self._should_end_run = True
                else:
                    # 既にアクティブな実行がある場合は終了しない
                    self._should_end_run = False
            except Exception:
                # エラーが発生した場合は新しい実行を開始
                mlflow.start_run(run_name=self.run_name)
                self._should_end_run = True
        else:
            # 新しい実行を開始しない
            self._should_end_run = False

        # 基本パラメータを記録
        mlflow.log_params({
            "outer_splits": self.outer_splits,
            "inner_splits": self.inner_splits,
            "n_trials": self.n_trials,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "tuning_method": "Optuna_TPE",
        })

        # パラメータ範囲を記録
        for param_name, param_range in self.param_ranges.items():
            mlflow.log_param(f"param_range_{param_name}", str(param_range))

    def _log_trial_results(self, trial: optuna.Trial, score: float, fold_idx: int = None) -> None:
        """Optunaの試行結果をMLflowに記録（現在は無効化）"""
        # MLflowには最終結果のみを記録するため、trial結果の記録は無効化
        pass

    def _log_fold_results(self, fold_idx: int, score: float, best_params: dict) -> None:
        """各フォールドの結果をMLflowに記録（現在は無効化）"""
        # MLflowには最終結果のみを記録するため、フォールド結果の記録は無効化
        pass

    def _objective(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series, fold_idx: int = None) -> float:
        """Optunaの目的関数 - 内側CVでパラメータを評価"""
        # パラメータをサンプリング
        params = {}
        for param_name, param_range in self.param_ranges.items():
            if isinstance(param_range, list):
                # カテゴリカルパラメータ
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # 数値パラメータ
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                raise ValueError(f"Unsupported parameter range format for {param_name}: {param_range}")

        # モデルを作成
        model = self.estimator_factory()
        model.set_params(**params)

        # 内側CVで評価
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, x_train, y_train, 
            cv=self.inner_splits, 
            scoring=self.scoring,
            n_jobs=1  # Optunaでは並列処理を避ける
        )
        
        score = scores.mean()
        
        return score

    def tune(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Optunaを使用してベイズ最適化でハイパーパラメータをチューニング"""
        print(f"🔄 Optunaベイズ最適化でチューニング中...")
        print(f"外側CV分割数: {self.outer_splits}")
        print(f"内側CV分割数: {self.inner_splits}")
        print(f"試行回数: {self.n_trials}")
        print(f"パラメータ範囲:")
        for param, range_val in self.param_ranges.items():
            print(f"  - {param}: {range_val}")

        # MLflowの設定
        self._setup_mlflow()

        from sklearn.model_selection import KFold
        
        outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        test_scores: list[float] = []
        best_params_list: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x), 1):
            print(f"🔄 外側CV {fold_idx}/{self.outer_splits} 回目のチューニング中...")
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Optunaスタディを作成
            study = optuna.create_study(
                direction="maximize" if self.scoring in ["r2", "accuracy"] else "minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )

            # ベイズ最適化を実行
            study.optimize(
                lambda trial: self._objective(trial, x_train, y_train, fold_idx),
                n_trials=self.n_trials,
                show_progress_bar=False
            )

            # 最適パラメータでモデルを作成
            best_model = self.estimator_factory()
            best_model.set_params(**study.best_params)
            best_model.fit(x_train, y_train)

            # テストスコアを計算
            score = best_model.score(x_test, y_test)
            test_scores.append(score)
            best_params_list.append(study.best_params)
            
            print(f"   ✅ 外側CV {fold_idx} 回目完了 - スコア: {score:.4f}")
            print(f"   🎯 最適パラメータ: {study.best_params}")

        avg_score = sum(test_scores) / len(test_scores)
        std_score = pd.Series(test_scores).std()
        print(f"🔧 平均スコア ({self.scoring}): {avg_score:.4f}")
        print(f"📊 スコアの標準偏差: {std_score:.4f}")
        
        # 最も良いスコアのパラメータを返す
        best_fold_idx = test_scores.index(max(test_scores))
        best_params = best_params_list[best_fold_idx]
        
        # 全データで最終モデルを学習
        print(f"🚀 全データで最終モデルを学習中...")
        final_model = self.estimator_factory()
        final_model.set_params(**best_params)
        final_model.fit(x, y)
        print(f"✅ 最終モデルの学習完了")
        
        # MLflowに最終結果を記録
        if self.use_mlflow:
            mlflow.log_metrics({
                "avg_score": avg_score,
                "std_score": std_score,
                "best_fold_score": max(test_scores),
                "worst_fold_score": min(test_scores),
            })
            
            # 最適パラメータを記録
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            # 各フォールドのスコアを記録
            for i, score in enumerate(test_scores, 1):
                mlflow.log_metric(f"fold_{i}_final_score", score)
            
            # 最終モデルをアーティファクトとして保存
            import joblib
            model_path = "final_model.pkl"
            joblib.dump(final_model, model_path)
            mlflow.log_artifact(model_path, "model")
            
            # チューニング結果のサマリーを保存
            import json
            summary = {
                "best_params": best_params,
                "avg_score": avg_score,
                "std_score": std_score,
                "all_scores": test_scores,
                "all_params": best_params_list,
                "tuning_info": {
                    "outer_splits": self.outer_splits,
                    "inner_splits": self.inner_splits,
                    "n_trials": self.n_trials,
                    "scoring": self.scoring,
                    "random_state": self.random_state,
                }
            }
            summary_path = "tuning_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(summary_path, "summary")
            
            # MLflowの実行を終了
            if self._should_end_run:
                mlflow.end_run()
                print(f"📊 MLflowに結果を記録しました - 実験名: {self.experiment_name}, 実行名: {self.run_name}")
        
        return {
            "best_params": best_params, 
            "avg_score": avg_score,
            "std_score": std_score,
            "all_scores": test_scores,
            "all_params": best_params_list,
            "final_model": final_model,  # 全データで学習した最終モデル
            "mlflow_info": {
                "experiment_name": self.experiment_name,
                "run_name": self.run_name,
                "use_mlflow": self.use_mlflow
            } if self.use_mlflow else None
        }
