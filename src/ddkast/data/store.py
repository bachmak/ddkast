from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class DataStore(Protocol):
    def read(self, name: str) -> pd.DataFrame: ...
    def write(self, name: str, df: pd.DataFrame) -> None: ...


class ParquetStore:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def read(self, name: str) -> pd.DataFrame:
        path = self._base / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No data at {path}. Run the preceding pipeline stage first."
            )
        return pd.read_parquet(path)

    def write(self, name: str, df: pd.DataFrame) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._base / f"{name}.parquet")
