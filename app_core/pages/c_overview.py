"""预测概览页布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="overview-total-success"),
                        lg=6,
                        md=12,
                    ),
                    dbc.Col(
                        dcc.Graph(id="overview-sector-success"),
                        lg=6,
                        md=12,
                    ),
                ],
                className="gy-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="overview-timeline-trend"),
                        lg=12,
                        md=12,
                    ),
                ],
                className="gy-4",
            ),
        ]
    )


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
