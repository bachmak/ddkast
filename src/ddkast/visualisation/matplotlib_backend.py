# pyright: reportUnknownMemberType = false
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from ddkast.config import Config
from ddkast.visualisation.protocol import VisualisationData

_ACTUAL_COLOR = "#2ca02c"
_FORECAST_COLOR = "#1f77b4"


class MatplotlibBackend:
    def render(self, data: VisualisationData, config: Config) -> list[str]:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        )
        ax_top = axes[0]  # type: ignore[index]
        ax_bot = axes[1]  # type: ignore[index]

        # --- top panel: actual vs forecast ---
        ax_top.plot(
            data.actual.index,
            data.actual.values,
            label="Actual",
            color=_ACTUAL_COLOR,
            linewidth=0.8,
        )
        if "forecast" in config.plots:
            ax_top.plot(
                data.forecast.index,
                data.forecast.values,
                label="Model forecast",
                color=_FORECAST_COLOR,
                linewidth=0.8,
            )
        ax_top.set_ylabel("Load (MW)")
        ax_top.legend(loc="upper right", fontsize=8)
        ax_top.grid(axis="y", linewidth=0.4, alpha=0.5)

        # --- bottom panel: residuals ---
        if "residuals" in config.plots:
            ax_bot.plot(
                data.residuals_forecast.index,
                data.residuals_forecast.values,
                color=_FORECAST_COLOR,
                linewidth=0.8,
            )
            ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
            ax_bot.set_ylabel("Residuals (MW)")
            ax_bot.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            ax_bot.set_visible(False)

        fig.autofmt_xdate(rotation=30, ha="right")

        config.plots_dir.mkdir(parents=True, exist_ok=True)
        out = config.plots_dir / f"forecast_analysis.{config.figure_format}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return [out.resolve().as_uri()]
