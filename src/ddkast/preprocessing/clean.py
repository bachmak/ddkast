from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Curate raw load data: resample, detect outliers, reject incomplete windows."""
    raise NotImplementedError
