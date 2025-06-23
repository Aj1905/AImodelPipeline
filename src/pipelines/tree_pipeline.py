"""
ツリーベースモデル用パイプライン。

LightGBMなどのツリーベースモデルに特化したパイプライン実装。
"""

import polars as pl

from ..features import FeatureManager
from ..evaluation.metrics.metrics import Metrics, evaluate_regression
from ..models.base import BaseModel
from ..evaluation.results.results import CrossValidationResult, TrainingResult
from ..features.managers.target_manager import TargetManager
from .base import BasePipeline


class TreeModelPipeline(BasePipeline):
    """ツリーベースモデル用の学習パイプライン。"""

    def __init__(self, model: BaseModel, target_manager: TargetManager, feature_manager: FeatureManager):
        """TreeModelPipelineを初期化する。

        Args:
            model (BaseModel): 使用するモデル
            target_manager (TargetManager): ターゲット管理クラス.
            feature_manager (FeatureManager): 特徴量管理クラス.
        """
        super().__init__(model, target_manager, feature_manager)

    def train(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        time_series_split: bool = False,
        time_column: str | None = None,
    ) -> TrainingResult:
        """事前に読み込まれたデータを使ってモデルを学習する。

        Args:
            test_size (float, optional): テストデータの割合. デフォルトは 0.2.
            random_state (int, optional): ランダムシード. デフォルトは 42.
            time_series_split (bool, optional): 時系列分割を使用するかどうか. デフォルトは False.
            time_column (str, optional): 時系列分割に使用する日付列名. time_series_split=Trueの場合必須.

        Returns:
            TrainingResult: 学習結果

        Raises:
            ValueError: データが読み込まれていない場合
            ValueError: test_sizeが無効な範囲の場合
            ValueError: データが不十分な場合
            ValueError: time_series_split=Trueでtime_columnが指定されていない場合
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_sizeは0.0より大きく1.0より小さい値である必要があります")

        # データ数をチェック
        total_data_count = len(self._target_manager)
        if total_data_count < 2:
            raise ValueError("データが不十分です。最低2つのデータポイントが必要です。")

        # train_test_splitの結果、訓練データが空になる場合をチェック
        n_test = int(total_data_count * test_size)
        n_train = total_data_count - n_test
        if n_train == 0:
            raise ValueError("データが不十分です。test_sizeが大きすぎて訓練データが空になります。")

        if time_series_split:
            if time_column is None:
                raise ValueError("時系列分割を使用する場合、time_columnを指定する必要があります。")
            x_train, x_test, y_train, y_test = self._time_series_train_test_split(test_size, time_column)
        else:
            x_train, x_test, y_train, y_test = self._train_test_split(test_size, random_state)

        x_train = x_train.drop(self._target_manager.get_target_name())
        x_test = x_test.drop(self._target_manager.get_target_name())
        if time_series_split and time_column is not None:
            x_train = x_train.drop(time_column)
            x_test = x_test.drop(time_column)

        self._model.train(x_train, y_train)

        y_train_pred = self._model.predict(x_train)
        y_test_pred = self._model.predict(x_test)

        train_metrics = evaluate_regression(y_train, y_train_pred)
        validation_metrics = evaluate_regression(y_test, y_test_pred)

        return TrainingResult(
            model=self._model,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            train_size=len(x_train),
            test_size=len(x_test),
            feature_count=len(x_train.columns),
        )

    def cross_validate(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> CrossValidationResult:
        """事前に読み込まれたデータでクロスバリデーションを実行する。

        Args:
            cv_folds (int, optional): クロスバリデーションの分割数. デフォルトは 5.
            random_state (int, optional): ランダムシード. デフォルトは 42.

        Returns:
            CrossValidationResult: クロスバリデーション結果

        Raises:
            ValueError: データが読み込まれていない場合
        """
        if cv_folds < 2:
            raise ValueError("クロスバリデーションの分割数は2以上である必要があります")

        if len(self._feature_manager) == 0:
            raise ValueError("特徴量データが読み込まれていません。load_data()を先に呼び出してください。")
        if len(self._target_manager) == 0:
            raise ValueError("ターゲットデータが読み込まれていません。load_data()を先に呼び出してください。")

        x = self._feature_manager.to_polars_dataframe(use_enabled_only=True)
        y = self._target_manager.to_polars_series()
        data = x.with_columns(y)
        target_column = self._target_manager.get_target_name()

        shuffled_data = data.sample(fraction=1.0, seed=random_state)
        fold_size = len(shuffled_data) // cv_folds

        fold_metrics = []

        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else len(shuffled_data)

            val_data = shuffled_data.slice(start_idx, end_idx - start_idx)
            train_data = (
                pl.concat([shuffled_data.head(start_idx), shuffled_data.tail(len(shuffled_data) - end_idx)])
                if start_idx > 0 or end_idx < len(shuffled_data)
                else pl.DataFrame()
            )

            if train_data.is_empty():
                continue

            x_train = train_data.drop(target_column)
            x_val = val_data.drop(target_column)
            y_train = train_data[target_column]
            y_val = val_data[target_column]

            self._model.train(x_train, y_train)

            y_val_pred = self._model.predict(x_val)
            metrics = evaluate_regression(y_val, y_val_pred)
            fold_metrics.append(metrics)

        mean_metrics = Metrics(mse=0.0, rmse=0.0, mae=0.0, r2=0.0)
        std_metrics = Metrics(mse=0.0, rmse=0.0, mae=0.0, r2=0.0)

        if fold_metrics:
            mse_values = [m.mse for m in fold_metrics]
            rmse_values = [m.rmse for m in fold_metrics]
            mae_values = [m.mae for m in fold_metrics]
            r2_values = [m.r2 for m in fold_metrics]

            mean_metrics = Metrics(
                mse=sum(mse_values) / len(mse_values),
                rmse=sum(rmse_values) / len(rmse_values),
                mae=sum(mae_values) / len(mae_values),
                r2=sum(r2_values) / len(r2_values),
            )

            std_metrics = Metrics(
                mse=(sum((v - mean_metrics.mse) ** 2 for v in mse_values) / len(mse_values)) ** 0.5,
                rmse=(sum((v - mean_metrics.rmse) ** 2 for v in rmse_values) / len(rmse_values)) ** 0.5,
                mae=(sum((v - mean_metrics.mae) ** 2 for v in mae_values) / len(mae_values)) ** 0.5,
                r2=(sum((v - mean_metrics.r2) ** 2 for v in r2_values) / len(r2_values)) ** 0.5,
            )

        return CrossValidationResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            cv_folds=cv_folds,
        )

    def _train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """データを学習用と検証用に分割する。

        Args:
            data (pl.DataFrame): 分割するデータ
            target_column (str): ターゲット列名
            test_size (float, optional): テストデータの割合. デフォルトは 0.2.
            random_state (int, optional): ランダムシード. デフォルトは 42.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
                (train_data, test_data, y_train, y_test)

        Raises:
            ValueError: ターゲット列が存在しない場合
        """

        x = self._feature_manager.to_polars_dataframe(use_enabled_only=True)
        y = self._target_manager.to_polars_series()

        data = x.with_columns(y)
        shuffled_data = data.sample(fraction=1.0, seed=random_state)

        n_test = int(len(shuffled_data) * test_size)
        n_train = len(shuffled_data) - n_test

        train_data = shuffled_data.head(n_train)
        test_data = shuffled_data.tail(n_test)

        target_column = self._target_manager.get_target_name()
        y_train = train_data[target_column]
        y_test = test_data[target_column]

        return train_data, test_data, y_train, y_test

    def _time_series_train_test_split(
        self,
        test_size: float = 0.2,
        time_column: str = "date",
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """時系列データを学習用と検証用に分割する

        Args:
            test_size (float, optional): テストデータの割合. デフォルトは 0.2.
            time_column (str, optional): ソートに使用する時系列列名. デフォルトは "date".

        Returns:
            tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
                (train_data, test_data, y_train, y_test)
        """
        x = self._feature_manager.to_polars_dataframe(use_enabled_only=True)
        y = self._target_manager.to_polars_series()

        # データを結合
        data = x.with_columns(y)

        if time_column not in data.columns:
            raise ValueError(f"指定された時系列列 '{time_column}' がデータに存在しません。")

        # 時系列列でソート (昇順)
        sorted_data = data.sort(time_column)

        # ユニークな値を取得 (順序を保持)
        unique_values = sorted_data[time_column].unique(maintain_order=True)
        n_unique = len(unique_values)

        # テストセットに含めるユニーク値の数を計算
        n_test_unique = max(1, int(n_unique * test_size))
        n_train_unique = n_unique - n_test_unique

        # 訓練用とテスト用の値を分割
        train_values = unique_values.head(n_train_unique)
        test_values = unique_values.tail(n_test_unique)

        # 値に基づいてデータを分割
        train_data = sorted_data.filter(pl.col(time_column).is_in(train_values.to_list()))
        test_data = sorted_data.filter(pl.col(time_column).is_in(test_values.to_list()))
        if train_data.is_empty() or test_data.is_empty():
            raise ValueError("データが不十分です。test_sizeが大きすぎて訓練データまたはテストデータが空になります。")

        target_column = self._target_manager.get_target_name()
        y_train = train_data[target_column]
        y_test = test_data[target_column]

        return train_data, test_data, y_train, y_test
