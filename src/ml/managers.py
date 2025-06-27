from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class FeatureManager:
    """シンプルな特徴量管理クラス"""

    features: pd.DataFrame | None = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.features = df.copy()
        return self.features


@dataclass
class TargetManager:
    """シンプルなターゲット管理クラス"""

    target: pd.Series | None = None

    def transform(self, series: pd.Series) -> pd.Series:
        self.target = series.copy()
        return self.target
