"""原始数据校验页布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import Dash, Input, Output, ctx, dcc, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            dcc.Interval(
                id="raw-data-trigger",
                interval=1500,
                n_intervals=0,
                max_intervals=1,
            ),
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H4("原始数据校验", className="mb-3"),
                            html.P(
                                "展示标的中从数据源直接获取的原始因子输入，便于核对抓取是否准确。",
                                className="text-muted",
                            ),
                            dbc.Button(
                                "重新加载原始数据",
                                id="raw-data-refresh",
                                color="primary",
                                size="sm",
                                className="me-2",
                            ),
                            html.Span(
                                "手动刷新会强制从配置的数据源重新拉取最新原始数据。",
                                className="text-muted",
                            ),
                            html.Hr(),
                            html.Div(
                                id="raw-data-status",
                                className="mb-3 fw-semibold",
                                style={"whiteSpace": "pre-wrap"},
                            ),
                            dag.AgGrid(
                                id="raw-data-table",
                                columnDefs=[
                                    {
                                        "headerName": "标的",
                                        "field": "symbol",
                                        "pinned": "left",
                                        "width": 90,
                                    },
                                    {
                                        "headerName": "时点",
                                        "field": "timeline",
                                        "width": 150,
                                    },
                                    {
                                        "headerName": "时点键",
                                        "field": "timeline_key",
                                        "width": 120,
                                    },
                                    {
                                        "headerName": "因子",
                                        "field": "factor",
                                        "width": 160,
                                    },
                                    {
                                        "headerName": "Z 值",
                                        "field": "z",
                                        "width": 90,
                                    },
                                    {
                                        "headerName": "标准化值",
                                        "field": "value",
                                        "width": 120,
                                    },
                                    {
                                        "headerName": "Median",
                                        "field": "median",
                                        "width": 110,
                                    },
                                    {
                                        "headerName": "MAD",
                                        "field": "mad",
                                        "width": 110,
                                    },
                                    {
                                        "headerName": "Raw 数据",
                                        "field": "raw",
                                        "minWidth": 220,
                                        "flex": 2,
                                    },
                                    {
                                        "headerName": "数据时间",
                                        "field": "as_of",
                                        "width": 160,
                                    },
                                    {
                                        "headerName": "来源",
                                        "field": "source",
                                        "width": 180,
                                    },
                                    {
                                        "headerName": "附加信息",
                                        "field": "details",
                                        "minWidth": 220,
                                        "flex": 1,
                                    },
                                ],
                                rowData=[],
                                defaultColDef={
                                    "sortable": True,
                                    "filter": True,
                                    "resizable": True,
                                    "wrapText": True,
                                    "autoHeight": True,
                                },
                                dashGridOptions={
                                    "domLayout": "autoHeight",
                                    "pagination": True,
                                    "paginationPageSize": 20,
                                },
                                style={"width": "100%"},
                                className="ag-theme-alpine",
                            ),
                            html.Div(
                                id="raw-data-summary",
                                className="mt-3 text-muted small",
                                style={"whiteSpace": "pre-wrap"},
                            ),
                        ]
                    )
                ],
                className="shadow-sm",
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 原始数据校验页的回调。"""

    @app.callback(
        Output("raw-data-status", "children"),
        Output("raw-data-table", "rowData"),
        Output("raw-data-summary", "children"),
        Input("raw-data-trigger", "n_intervals"),
        Input("raw-data-refresh", "n_clicks"),
        prevent_initial_call=False,
    )
    def render_data_source_audit(n_intervals, refresh_clicks):  # noqa: D401
        del n_intervals, refresh_clicks
        force_reload = ctx.triggered_id == "raw-data-refresh"
        status, rows, summary = core.build_data_source_audit(force_reload=force_reload)
        return status, rows, summary
