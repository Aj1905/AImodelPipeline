from typing import Any, Callable, Dict, List

import pandas as pd

from .nested_cv import nested_cv_evaluate


class HyperTuner:
    """ネスト化クロスバリデーションを用いたハイパーパラメータチューニングクラス"""

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        param_grid: Dict[str, List[Any]],
        outer_splits: int = 5,
        inner_splits: int = 3,
        scoring: str = "r2",
    ) -> None:
        self.estimator_factory = estimator_factory
        self.param_grid = param_grid
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.scoring = scoring

    def tune(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """ネストCVにより最適パラメータを探索し、平均スコアを返す"""
        print(f"🔄 Nested CV ({self.outer_splits}×{self.inner_splits}) でチューニング中...")
        scores, best_params = nested_cv_evaluate(
            estimator_factory=self.estimator_factory,
            param_grid=self.param_grid,
            X=X,
            y=y,
            outer_splits=self.outer_splits,
            inner_splits=self.inner_splits,
            scoring=self.scoring,
        )
        avg_score = sum(scores) / len(scores)
        print(f"🔧 平均スコア ({self.scoring}): {avg_score:.4f}")
        return {"best_params": best_params[0], "avg_score": avg_score}
