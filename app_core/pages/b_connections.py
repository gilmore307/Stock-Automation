"""数据连接状态页面的布局与回调。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:
    """Return the tab contents for the connection screen."""

    return html.Div(
        [
            html.Div(
                "请确认 Firstrade 登录成功后，再使用下方按钮检查当前的账号与数据连接状态。",
                style={"marginBottom": "12px", "whiteSpace": "pre-wrap"},
            ),
            html.Div(
                [
                    dbc.Button("检查连接", id="check-connections-btn", color="info", size="sm"),
                    html.Span(
                        "系统会每分钟自动巡检连接状态，您也可以手动点击按钮立即更新。",
                        style={"marginLeft": "12px", "fontSize": "0.9rem"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "8px"},
            ),
            html.Div(
                id="connection-status-summary",
                style={"marginTop": "12px", "whiteSpace": "pre-wrap", "fontWeight": "600"},
            ),
            dag.AgGrid(
                id="connection-status-grid",
                columnDefs=[
                    {"headerName": "检查项", "field": "parameter", "minWidth": 160},
                    {"headerName": "状态", "field": "status", "maxWidth": 100},
                    {"headerName": "详情", "field": "detail", "minWidth": 220, "flex": 1},
                    {"headerName": "原始数据", "field": "raw", "minWidth": 260, "flex": 1},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": False,
                    "resizable": True,
                    "filter": False,
                    "wrapText": True,
                    "autoHeight": True,
                },
                dashGridOptions={
                    "treeData": True,
                    "animateRows": False,
                    "groupDefaultExpanded": 0,
                    "getDataPath": {"function": "params.data.path"},
                    "autoGroupColumnDef": {
                        "headerName": "资源 / 明细",
                        "minWidth": 220,
                        "cellRendererParams": {"suppressCount": True},
                    },
                },
                className="ag-theme-alpine",
                style={"width": "100%", "marginTop": "12px"},
            ),
            html.Div(
                "示例原始数据使用 AAPL 作为采样标的。",
                className="text-muted",
                style={"marginTop": "8px", "fontSize": "0.85rem"},
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register callbacks for连接巡检页面."""

    @app.callback(
        Output("connection-status-summary", "children"),
        Output("connection-status-grid", "rowData"),
        Input("check-connections-btn", "n_clicks"),
        Input("connection-poller", "n_intervals"),
        State("ft-session-store", "data"),
    )
    def refresh_connection_status(n_clicks, poll_intervals, ft_session):  # noqa: D401
        del n_clicks, poll_intervals
        session_data = ft_session if isinstance(ft_session, dict) else None
        summary, rows = core.build_resource_connection_overview(session_data)
        return summary, rows
