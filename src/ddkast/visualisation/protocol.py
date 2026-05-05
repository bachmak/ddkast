from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ddkast.config import Config


@dataclass
class VisualisationData:
    actual: pd.Series[float]
    forecast: pd.Series[float]
    entso_daf: pd.Series[float]
    residuals_forecast: pd.Series[float]
    residuals_daf: pd.Series[float] | None


class PlotBackend(Protocol):
    def render(self, data: VisualisationData, config: Config) -> list[str]: ...
