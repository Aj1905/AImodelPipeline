"""
回帰モデルの評価指標。

回帰モデルで一般的に使用される評価指標を提供します。
"""

from dataclasses import dataclass

import polars as pl


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        """評価指標の見やすい文字列表現を返す。"""
        return f"MSE: {self.mse:.4f}, RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}, R²: {self.r2:.4f}"


def mean_squared_error(y_true: pl.Series, y_pred: pl.Series) -> float:
    """平均二乗誤差(MSE)を計算する。

    Args:
        y_true (pl.Series): 実際の値
        y_pred (pl.Series): 予測値

    Returns:
        float: MSE値

    Raises:
        ValueError: 入力データの長さが一致しない場合
    """
    if len(y_true) != len(y_pred):
        raise ValueError("実際の値と予測値の長さが一致しません")

    if len(y_true) == 0:
        raise ValueError("入力データが空です")

    diff = y_true - y_pred
    mse = (diff * diff).mean()
    return float(mse)


def root_mean_squared_error(y_true: pl.Series, y_pred: pl.Series) -> float:
    """平均平方根誤差(RMSE)を計算する。

    Args:
        y_true (pl.Series): 実際の値
        y_pred (pl.Series): 予測値

    Returns:
        float: RMSE値

    Raises:
        ValueError: 入力データの長さが一致しない場合
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse**0.5


def mean_absolute_error(y_true: pl.Series, y_pred: pl.Series) -> float:
    """平均絶対誤差(MAE)を計算する。

    Args:
        y_true (pl.Series): 実際の値
        y_pred (pl.Series): 予測値

    Returns:
        float: MAE値

    Raises:
        ValueError: 入力データの長さが一致しない場合
    """
    if len(y_true) != len(y_pred):
        raise ValueError("実際の値と予測値の長さが一致しません")

    if len(y_true) == 0:
        raise ValueError("入力データが空です")

    diff = y_true - y_pred
    mae = diff.abs().mean()
    return float(mae)


def r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """決定係数(R²)を計算する。

    Args:
        y_true (pl.Series): 実際の値
        y_pred (pl.Series): 予測値

    Returns:
        float: R²値

    Raises:
        ValueError: 入力データの長さが一致しない場合
    """
    if len(y_true) != len(y_pred):
        raise ValueError("実際の値と予測値の長さが一致しません")

    if len(y_true) == 0:
        raise ValueError("入力データが空です")

    y_mean = y_true.mean()
    ss_tot = ((y_true - y_mean) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def evaluate_regression(y_true: pl.Series, y_pred: pl.Series) -> Metrics:
    """回帰モデルの包括的な評価を実行する。

    Args:
        y_true (pl.Series): 実際の値
        y_pred (pl.Series): 予測値

    Returns:
        dict[str, float]: 各評価指標の結果

    Raises:
        ValueError: 入力データの長さが一致しない場合
    """
    return Metrics(
        mse=mean_squared_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
    )
