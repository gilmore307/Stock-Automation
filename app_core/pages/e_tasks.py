"""任务中心相关布局与回调。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
from dash import Dash, Input, Output, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            dag.AgGrid(
                id="task-table",
                columnDefs=[
                    {"headerName": "任务", "field": "name"},
                    {"headerName": "状态", "field": "status"},
                    {"headerName": "标的进度", "field": "symbol_progress"},
                    {"headerName": "详情", "field": "detail"},
                    {"headerName": "开始时间", "field": "start_time"},
                    {"headerName": "结束时间", "field": "end_time"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": False,
                    "resizable": True,
                    "filter": False,
                    "flex": 1,
                    "minWidth": 140,
                },
                dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                style={"width": "100%"},
                className="ag-theme-alpine",
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 任务中心的渲染回调。"""

    @app.callback(
        Output("task-table", "rowData"),
        Input("task-store", "data"),
    )
    def render_task_table(task_state):  # noqa: D401
        return core.render_task_table_logic(task_state)
