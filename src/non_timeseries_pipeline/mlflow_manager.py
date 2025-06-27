from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import joblib
import mlflow


class MLflowManager:
    """Handle MLflow logging for hyperparameter tuning."""

    def __init__(self, experiment_name: str, run_name: str | None, start_run: bool) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.start_run = start_run
        self._should_end_run = False

    def setup(
        self,
        outer_splits: int,
        inner_splits: int,
        n_trials: int,
        scoring: str,
        random_state: int,
        param_ranges: dict[str, Any],
    ) -> None:
        """Set up MLflow experiment and start run if needed."""
        mlflow.set_experiment(self.experiment_name)
        if not self.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"optuna_tuning_{timestamp}"
        if self.start_run:
            current_run = mlflow.active_run()
            if current_run is None:
                mlflow.start_run(run_name=self.run_name)
                self._should_end_run = True
        mlflow.log_params(
            {
                "outer_splits": outer_splits,
                "inner_splits": inner_splits,
                "n_trials": n_trials,
                "scoring": scoring,
                "random_state": random_state,
                "tuning_method": "Optuna_TPE",
            }
        )
        for name, rng in param_ranges.items():
            mlflow.log_param(f"param_range_{name}", str(rng))

    def log_results(
        self,
        best_params: dict[str, Any],
        avg_score: float,
        std_score: float,
        test_scores: list[float],
        final_model: Any,
    ) -> None:
        """Log tuning results and artifacts."""
        mlflow.log_metrics(
            {
                "avg_score": avg_score,
                "std_score": std_score,
                "best_fold_score": max(test_scores),
                "worst_fold_score": min(test_scores),
            }
        )
        for name, value in best_params.items():
            mlflow.log_param(f"best_{name}", value)
        for i, score in enumerate(test_scores, 1):
            mlflow.log_metric(f"fold_{i}_final_score", score)
        model_path = "final_model.pkl"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(model_path, "model")
        summary = {
            "best_params": best_params,
            "avg_score": avg_score,
            "std_score": std_score,
            "all_scores": test_scores,
        }
        summary_path = "tuning_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(summary_path, "summary")

    def end_run(self) -> None:
        """End MLflow run if this manager started it."""
        if self._should_end_run:
            mlflow.end_run()
            print(f"\n📊 MLflowに結果を記録しました - 実験名: {self.experiment_name}, 実行名: {self.run_name}")
