from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold


def nested_cv_evaluate(
    estimator_factory: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    x: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "r2",
) -> tuple[list[float], list[dict[str, Any]]]:
    """Evaluate model hyperparameters via nested cross validation."""
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    test_scores: list[float] = []
    best_params_list: list[dict[str, Any]] = []

    for train_idx, test_idx in outer_cv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=estimator_factory(),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        best_model = grid.best_estimator_
        score = best_model.score(x_test, y_test)

        test_scores.append(score)
        best_params_list.append(grid.best_params_)

    return test_scores, best_params_list
