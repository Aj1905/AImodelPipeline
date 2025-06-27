from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureManager:
    """Manage feature transformations"""

    features: pd.DataFrame | None = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Store and return the given features."""
        self.features = df.copy()
        return self.features
