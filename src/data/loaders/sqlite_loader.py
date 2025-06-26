from pathlib import Path

import pandas as pd

from ..handlers.sqlite_handler import SQLiteHandler


class SQLiteDataLoader:
    """Load data from SQLite database."""

    def load_columns(self, db_path: Path, table_name: str, columns: list[str]) -> pd.DataFrame:
        """Return selected columns as DataFrame."""
        handler = SQLiteHandler(db_path)
        quoted = [f'"{c}"' for c in columns]
        query = f"SELECT {', '.join(quoted)} FROM {table_name}"
        try:
            results = handler.fetch_all(query)
            if not results:
                return pd.DataFrame()
            return pd.DataFrame(results, columns=columns)
        except Exception as e:  # pragma: no cover - simple print fallback
            print(f"❌ データ読み込みエラー: {e}")
            return pd.DataFrame()
