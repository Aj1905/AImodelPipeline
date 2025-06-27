from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold


def nested_cv_evaluate(
    estimator_factory: Callable[[], Any],
    param_grid: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "r2",
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """外側CVと内側CVを組み合わせた評価関数"""
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    test_scores: List[float] = []
    best_params_list: List[Dict[str, Any]] = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=estimator_factory(),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        score = best_model.score(X_test, y_test)

        test_scores.append(score)
        best_params_list.append(grid.best_params_)

    return test_scores, best_params_list
