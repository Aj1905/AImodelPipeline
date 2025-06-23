"""
機械学習モデルの基底クラス。

すべてのモデル実装の基底となるクラスを定義します。
"""

from pathlib import Path
from typing import Any

import polars as pl

from ..interfaces.model_interface import IModel


class BaseModel(IModel):
    """機械学習モデルの基底クラス。"""

    def __init__(self, model_name: str | None = None):
        """BaseModelを初期化する。

        Args:
            model_name (str | None, optional): モデルの名称. デフォルトは None.
        """
        self.model_name = model_name or self.__class__.__name__
        self._model: Any = None
        self._is_trained: bool = False

    def is_trained(self) -> bool:
        """モデルが学習済みかどうかを確認する。

        Returns:
            bool: 学習済みの場合は True、そうでない場合は False
        """
        return self._is_trained

    def get_model_name(self) -> str:
        """モデル名を取得する。

        Returns:
            str: モデル名
        """
        return self.model_name

    def set_model_name(self, name: str) -> None:
        """モデル名を設定する。

        Args:
            name (str): 設定するモデル名
        """
        self.model_name = name

    def __repr__(self) -> str:
        """オブジェクトの文字列表現を返す。

        Returns:
            str: オブジェクトの文字列表現
        """
        status = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')" 