"""财报日程页面与后台抓取的 Dash 回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
from dash import Dash, Input, Output, State, dcc, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:
    """Return layout for the earnings schedule tab."""

    return html.Div(
        [
            html.Div(
                [
                    html.Label("选择决策日"),
                    dcc.DatePickerSingle(
                        id="earnings-date-picker",
                        date=config.default_picker_date_str,
                        min_date_allowed=config.min_picker_date,
                        max_date_allowed=config.max_picker_date,
                        display_format="YYYY-MM-DD",
                        initial_visible_month=config.default_picker_date,
                        day_size=36,
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Button("刷新列表", id="earnings-refresh-btn"),
                    html.Button("下载 CSV", id="dl-btn"),
                ],
                style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
            ),
            html.Div(id="status", style={"margin": "8px 0", "whiteSpace": "pre-wrap"}),
            dag.AgGrid(
                id="table",
                columnDefs=[
                    {"headerName": "代码", "field": "symbol", "filter": "agTextColumnFilter"},
                    {"headerName": "公司", "field": "company", "filter": "agTextColumnFilter"},
                    {"headerName": "决策日期", "field": "decision_date", "filter": "agDateColumnFilter"},
                    {"headerName": "时间段", "field": "bucket", "filter": "agTextColumnFilter"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "flex": 1,
                    "minWidth": 120,
                },
                dashGridOptions={
                    "domLayout": "autoHeight",
                    "pagination": True,
                    "paginationPageSize": 20,
                    "animateRows": True,
                    "rowSelection": "multiple",
                },
                style={"height": "600px", "width": "100%"},
                className="ag-theme-alpine",
            ),
            dcc.Download(id="dl"),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 财报日程页相关回调。"""

    @app.callback(
        Output("log-store", "data", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Output("table", "rowData", allow_duplicate=True),
        Output("task-store", "data", allow_duplicate=True),
        Input("auto-run-trigger", "n_intervals"),
        Input("earnings-date-picker", "date"),
        Input("earnings-refresh-btn", "n_clicks"),
        Input("ft-session-store", "data"),
        State("task-store", "data"),
        State("log-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def start_run(auto_intervals, selected_date, refresh_clicks, session_data, task_state, log_state):  # noqa: D401
        return core.start_run_logic(
            auto_intervals,
            selected_date,
            refresh_clicks,
            session_data,
            task_state,
            log_state,
        )

    @app.callback(
        Output("dl", "data"),
        Input("dl-btn", "n_clicks"),
        State("table", "rowData"),
        prevent_initial_call=True,
    )
    def download_csv(n, data):
        return core.download_csv_logic(n, data)
