# pyright: reportUnknownMemberType = false
from __future__ import annotations

import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ddkast.config import Config
from ddkast.visualisation.protocol import VisualisationData

_ACTUAL_COLOR = "#2ca02c"
_FORECAST_COLOR = "#1f77b4"
_DAF_COLOR = "#ff7f0e"
_ZERO_COLOR = "rgba(128,128,128,0.6)"


class PlotlyBackend:
    def render(self, data: VisualisationData, config: Config) -> list[str]:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=("Load (MW)", "Residuals (MW)"),
        )

        # --- top panel ---
        fig.add_trace(
            go.Scatter(
                x=data.actual.index,
                y=data.actual.values,
                name="Actual",
                line={"color": _ACTUAL_COLOR, "width": 1},
            ),
            row=1,
            col=1,
        )

        if "forecast" in config.plots:
            fig.add_trace(
                go.Scatter(
                    x=data.forecast.index,
                    y=data.forecast.values,
                    name="Model forecast",
                    line={"color": _FORECAST_COLOR, "width": 1},
                ),
                row=1,
                col=1,
            )

        if "daf" in config.plots and not data.entso_daf.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=data.entso_daf.index,
                    y=data.entso_daf.values,
                    name="ENTSO-E DAF",
                    line={"color": _DAF_COLOR, "width": 1, "dash": "dot"},
                ),
                row=1,
                col=1,
            )

        # --- bottom panel ---
        if "residuals" in config.plots:
            if "forecast" in config.plots:
                fig.add_trace(
                    go.Scatter(
                        x=data.residuals_forecast.index,
                        y=data.residuals_forecast.values,
                        name="Model residuals",
                        line={"color": _FORECAST_COLOR, "width": 1},
                    ),
                    row=2,
                    col=1,
                )

            if "daf" in config.plots and data.residuals_daf is not None:
                fig.add_trace(
                    go.Scatter(
                        x=data.residuals_daf.index,
                        y=data.residuals_daf.values,
                        name="DAF residuals",
                        line={"color": _DAF_COLOR, "width": 1, "dash": "dot"},
                    ),
                    row=2,
                    col=1,
                )

            # Zero reference line
            idx = data.residuals_forecast.index
            x_ends = [idx[0], idx[-1]]
            fig.add_trace(
                go.Scatter(
                    x=x_ends,
                    y=[0, 0],
                    mode="lines",
                    line={"color": _ZERO_COLOR, "width": 1, "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title="Energy Load Forecast Analysis",
            height=720,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            xaxis={"rangeslider": {"visible": False}},
            xaxis2={"rangeslider": {"visible": True, "thickness": 0.04}},
        )

        config.plots_dir.mkdir(parents=True, exist_ok=True)
        out = config.plots_dir / "forecast_analysis.html"
        fig.write_html(str(out))

        return [out.resolve().as_uri()]
