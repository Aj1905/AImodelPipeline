"""
LightGBMを使用した回帰モデル実装。

BaseModelを継承してLightGBMの回帰モデルを実装します。
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from .base import BaseModel


class LightGBMRegressor(BaseModel):
    """LightGBMを使用した回帰モデル。"""

    _model: lgb.Booster

    def __init__(
        self,
        model_name: str | None = None,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 100,
        early_stopping_rounds: int | None = None,
        verbose_eval: bool = False,
    ):
        """LightGBMRegressorを初期化する。

        Args:
            model_name (str | None, optional): モデルの名称. デフォルトは None.
            params (dict[str, Any] | None, optional): LightGBMのパラメータ. デフォルトは None.
            num_boost_round (int, optional): ブースティング回数. デフォルトは 100.
            early_stopping_rounds (int | None, optional): 早期停止のラウンド数. デフォルトは None.
            verbose_eval (bool, optional): 学習過程を表示するかどうか. デフォルトは False.
        """
        super().__init__(model_name)

        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self._feature_names: list[str] | None = None
        # 学習時の情報を保持するための属性
        self._training_info: dict[str, Any] = {}
        self._custom_comments: list[str] = []

    def train(self, features: pl.DataFrame, targets: pl.Series) -> None:
        """モデルを学習する。

        Args:
            features (pl.DataFrame): 特徴量データ
            targets (pl.Series): 正解データ

        Raises:
            ValueError: 特徴量またはターゲットが空の場合
        """
        if features.is_empty():
            raise ValueError("特徴量データが空です")

        if targets.is_empty():
            raise ValueError("ターゲットデータが空です")

        if features.shape[0] != len(targets):
            raise ValueError("特徴量とターゲットの行数が一致しません")

        # データ型の最終チェック
        print("\n🔍 LightGBM用データ型チェック:")
        for col in features.columns:
            dtype = features[col].dtype
            if dtype == pl.Utf8:
                raise ValueError(f"文字列型の列 '{col}' が残っています。数値型に変換してください。")
            print(f"  {col}: {dtype}")

        self._feature_names = features.columns

        x = features.to_numpy()
        y = targets.to_numpy()

        train_data = lgb.Dataset(x, label=y, feature_name=self._feature_names)

        callbacks = []
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=10))

        self._model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            callbacks=callbacks if callbacks else None,
        )

        # 学習時の情報を記録
        self._training_info = {
            "data_size": len(features),
            "feature_count": len(features.columns),
            "training_timestamp": datetime.now().isoformat(),
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self._model.best_iteration if hasattr(self._model, "best_iteration") else None,
            "best_score": self._model.best_score if hasattr(self._model, "best_score") else None,
        }

        self._is_trained = True

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """予測を実行する。

        Args:
            features (pl.DataFrame): 特徴量データ

        Returns:
            pl.Series: 予測結果

        Raises:
            RuntimeError: モデルが学習されていない場合
            ValueError: 特徴量の列数が学習時と異なる場合
        """
        if not self._is_trained:
            raise RuntimeError("モデルが学習されていません。先にtrainメソッドを実行してください")

        if features.is_empty():
            raise ValueError("特徴量データが空です")

        if self._feature_names and len(features.columns) != len(self._feature_names):
            raise ValueError(
                f"特徴量の列数が学習時と異なります。期待値: {len(self._feature_names)}, 実際: {len(features.columns)}"
            )

        x = features.to_numpy()

        predictions = self._model.predict(x)

        return pl.Series("predictions", predictions)

    def save_model(self, file_path: str | Path) -> None:
        """学習済みモデルを保存する。

        Args:
            file_path (str | Path): 保存先ファイルパス

        Raises:
            RuntimeError: モデルが学習されていない場合
        """
        if not self._is_trained:
            raise RuntimeError("モデルが学習されていません。先にtrainメソッドを実行してください")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "params": self.params,
            "feature_names": self._feature_names,
            "model_name": self.model_name,
            "training_info": self._training_info,
            "custom_comments": self._custom_comments,
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, file_path: str | Path) -> None:
        """学習済みモデルを読み込む。

        Args:
            file_path (str | Path): 読み込み元ファイルパス

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            Exception: モデル読み込み中にエラーが発生した場合
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {file_path}")

        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            self._model = model_data["model"]
            self.params = model_data["params"]
            self._feature_names = model_data["feature_names"]
            self.model_name = model_data["model_name"]
            self._training_info = model_data.get("training_info", {})
            self._custom_comments = model_data.get("custom_comments", [])
            self._is_trained = True

        except Exception as err:
            raise Exception("エラーが発生しました") from err

    def get_feature_importance(self) -> dict[str, float]:
        """特徴量重要度を取得する。

        Returns:
            dict[str, float]: 特徴量名と重要度の辞書

        Raises:
            RuntimeError: モデルが学習されていない場合
        """
        if not self._is_trained:
            raise RuntimeError("モデルが学習されていません。先にtrainメソッドを実行してください")

        importance = self._model.feature_importance(importance_type="gain")

        if self._feature_names:
            return dict(zip(self._feature_names, importance, strict=False))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}

    def get_params(self) -> dict[str, Any]:
        """モデルのパラメータを取得する。

        Returns:
            dict[str, Any]: モデルパラメータ
        """
        return self.params.copy()

    def get_training_info(self) -> dict[str, Any]:
        """学習時の情報を取得する。

        Returns:
            dict[str, Any]: 学習時の情報
        """
        return self._training_info.copy()

    def print_training_info(self) -> None:
        """学習時の情報を表示する。"""
        if not self._is_trained:
            print("⚠️  モデルが学習されていません")
            return

        print("\n📊 学習時の情報:")
        print(f"  モデル名: {self.model_name}")
        print(f"  データサイズ: {self._training_info.get('data_size', 'N/A')}")
        print(f"  特徴量数: {self._training_info.get('feature_count', 'N/A')}")
        print(f"  学習日時: {self._training_info.get('training_timestamp', 'N/A')}")
        print(f"  ブースティング回数: {self._training_info.get('num_boost_round', 'N/A')}")
        print(f"  早期停止回数: {self._training_info.get('early_stopping_rounds', 'N/A')}")
        print(f"  最適イテレーション: {self._training_info.get('best_iteration', 'N/A')}")

        best_score = self._training_info.get("best_score", {})
        if best_score:
            print("  最適スコア:")
            for metric, value in best_score.items():
                print(f"    {metric}: {value}")

        if self._custom_comments:
            print("  カスタムコメント:")
            for i, comment in enumerate(self._custom_comments, 1):
                print(f"    {i}. {comment}")

    def add_comment(self, comment: str) -> None:
        """カスタムコメントを追加する。

        Args:
            comment (str): 追加するコメント
        """
        self._custom_comments.append(comment)

    def get_comments(self) -> list[str]:
        """カスタムコメントを取得する。

        Returns:
            list[str]: カスタムコメントのリスト
        """
        return self._custom_comments.copy()
