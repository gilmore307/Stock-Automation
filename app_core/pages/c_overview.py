"""预测概览页布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, ctx, dcc, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            dcc.Interval(
                id="overview-preview-trigger",
                interval=1500,
                n_intervals=0,
                max_intervals=1,
            ),
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H4("预测任务概览", className="mb-3"),
                            html.P(
                                "展示最近一个标的的完整预测过程，包含数据采集与计算细节。",
                                className="text-muted",
                            ),
                            dbc.Button(
                                "刷新概览",
                                id="overview-preview-run",
                                color="primary",
                                size="sm",
                                className="me-2",
                            ),
                            html.Span(
                                "点击按钮可立即获取最新的任务进展与参数。",
                                className="text-muted",
                            ),
                            html.Hr(),
                            html.Div(
                                id="overview-preview-status",
                                className="mb-3 fw-semibold",
                                style={"whiteSpace": "pre-wrap"},
                            ),
                            html.Div(id="overview-preview-steps", className="mb-4"),
                            html.Div(id="overview-preview-table"),
                        ]
                    )
                ],
                className="shadow-sm",
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 概览页测试演示回调。"""

    @app.callback(
        Output("overview-preview-status", "children"),
        Output("overview-preview-steps", "children"),
        Output("overview-preview-table", "children"),
        Input("overview-preview-trigger", "n_intervals"),
        Input("overview-preview-run", "n_clicks"),
        Input("prediction-store", "data"),
        prevent_initial_call=False,
    )
    def render_prediction_preview(n_intervals, run_clicks, prediction_store):  # noqa: D401
        del n_intervals, run_clicks
        force_reload = ctx.triggered_id == "overview-preview-run"
        return core.build_prediction_preview(
            prediction_store,
            force_reload=force_reload,
        )
