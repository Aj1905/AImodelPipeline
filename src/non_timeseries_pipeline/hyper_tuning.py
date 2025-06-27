from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from .mlflow_manager import MLflowManager


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
        run_name: str | None = None,
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
        self.mlflow_manager: MLflowManager | None = None
        if self.use_mlflow:
            self.mlflow_manager = MLflowManager(
                experiment_name=self.experiment_name,
                run_name=self.run_name,
                start_run=self.start_mlflow_run,
            )

    def _setup_mlflow(self) -> None:
        """Set up MLflow using :class:`MLflowManager`."""
        if self.mlflow_manager is None:
            return
        self.mlflow_manager.setup(
            outer_splits=self.outer_splits,
            inner_splits=self.inner_splits,
            n_trials=self.n_trials,
            scoring=self.scoring,
            random_state=self.random_state,
            param_ranges=self.param_ranges,
        )

    def _log_trial_results(self, trial: optuna.Trial, score: float, fold_idx: int | None = None) -> None:
        """Log each Optuna trial result (disabled)."""
        # MLflowには最終結果のみを記録するため、trial結果の記録は無効化
        pass

    def _log_fold_results(self, fold_idx: int, score: float, best_params: dict) -> None:
        """Log each fold result (disabled)."""
        # MLflowには最終結果のみを記録するため、フォールド結果の記録は無効化
        pass

    def _objective(
        self,
        trial: optuna.Trial,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        fold_idx: int | None = None,
    ) -> float:
        """Objective function for Optuna using inner CV."""
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

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=self.inner_splits,
            scoring=self.scoring,
            n_jobs=1,  # Optunaでは並列処理を避ける
        )

        score = scores.mean()

        return score

    def tune(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Optunaを使用してベイズ最適化でハイパーパラメータをチューニング"""
        print("🔄 Optunaベイズ最適化でチューニング中...")
        print(f"外側CV分割数: {self.outer_splits}")
        print(f"内側CV分割数: {self.inner_splits}")
        print(f"試行回数: {self.n_trials}")
        print("パラメータ範囲:")
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
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            # ベイズ最適化を実行
            study.optimize(
                lambda trial, x_train=x_train, y_train=y_train, fold_idx=fold_idx: self._objective(
                    trial,
                    x_train,
                    y_train,
                    fold_idx,
                ),
                n_trials=self.n_trials,
                show_progress_bar=False,
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
        print("🚀 全データで最終モデルを学習中...")
        final_model = self.estimator_factory()
        final_model.set_params(**best_params)
        final_model.fit(x, y)
        print("✅ 最終モデルの学習完了")

        if self.mlflow_manager is not None:
            self.mlflow_manager.log_results(
                best_params=best_params,
                avg_score=avg_score,
                std_score=std_score,
                test_scores=test_scores,
                final_model=final_model,
            )
            self.mlflow_manager.end_run()

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
                "use_mlflow": self.use_mlflow,
            }
            if self.use_mlflow
            else None,
        }
