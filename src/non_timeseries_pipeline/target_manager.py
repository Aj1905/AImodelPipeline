from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TargetManager:
    """Manage target variable."""

    target: pd.Series | None = None

    def transform(self, series: pd.Series) -> pd.Series:
        """Store and return the given target series."""
        self.target = series.copy()
        return self.target
