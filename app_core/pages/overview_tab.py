"""预测概览页回调封装。"""

from __future__ import annotations

from dash import Dash, Input, Output

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register 概览页图表刷新回调。"""

    @app.callback(
        Output("overview-total-success", "figure"),
        Output("overview-sector-success", "figure"),
        Output("overview-timeline-trend", "figure"),
        Input("prediction-store", "data"),
        Input("evaluation-store", "data"),
    )
    def render_overview_charts(prediction_store, evaluation_store):  # noqa: D401
        return core.render_overview_charts_logic(prediction_store, evaluation_store)
