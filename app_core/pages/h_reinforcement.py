"""强化学习模型页面布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
from dash import Dash, Input, Output, State, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            html.Div(
                [
                    html.H5("全局模型概览"),
                    dag.AgGrid(
                        id="rl-model-table",
                        columnDefs=[
                            {"headerName": "模型", "field": "model"},
                            {"headerName": "参数", "field": "parameter"},
                            {"headerName": "取值", "field": "value"},
                            {"headerName": "用途说明", "field": "description"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "sortable": True,
                            "resizable": True,
                            "filter": True,
                            "flex": 1,
                            "minWidth": 140,
                        },
                        dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                        className="ag-theme-alpine",
                    ),
                    html.Div(
                        [
                            html.H6("参数变化追踪"),
                            dag.AgGrid(
                                id="model-global-history-table",
                                columnDefs=[
                                    {"headerName": "时间", "field": "timestamp", "minWidth": 160},
                                    {"headerName": "参数", "field": "parameter", "minWidth": 140},
                                    {"headerName": "旧值", "field": "old_value", "minWidth": 120},
                                    {"headerName": "新值", "field": "new_value", "minWidth": 120},
                                ],
                                rowData=[],
                                defaultColDef={
                                    "sortable": True,
                                    "resizable": True,
                                    "filter": True,
                                    "flex": 1,
                                    "minWidth": 120,
                                },
                                dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                                className="ag-theme-alpine",
                            ),
                        ],
                        style={"marginTop": "16px"},
                    ),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(
                [
                    html.H5("行业模型概览"),
                    dag.AgGrid(
                        id="model-sector-table",
                        columnDefs=[
                            {"headerName": "行业", "field": "sector"},
                            {"headerName": "预测次数", "field": "total_predictions"},
                            {"headerName": "更新次数", "field": "update_count"},
                            {"headerName": "当前基准", "field": "baseline"},
                            {"headerName": "主要权重", "field": "top_weights"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "sortable": True,
                            "resizable": True,
                            "filter": True,
                            "flex": 1,
                            "minWidth": 140,
                        },
                        dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                        className="ag-theme-alpine",
                    ),
                    html.Div(
                        [
                            html.H6("行业参数变化追踪"),
                            dag.AgGrid(
                                id="model-sector-history-table",
                                columnDefs=[
                                    {"headerName": "时间", "field": "timestamp", "minWidth": 160},
                                    {"headerName": "行业", "field": "sector", "minWidth": 120},
                                    {"headerName": "参数", "field": "parameter", "minWidth": 140},
                                    {"headerName": "旧值", "field": "old_value", "minWidth": 120},
                                    {"headerName": "新值", "field": "new_value", "minWidth": 120},
                                ],
                                rowData=[],
                                defaultColDef={
                                    "sortable": True,
                                    "resizable": True,
                                    "filter": True,
                                    "flex": 1,
                                    "minWidth": 120,
                                },
                                dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                                className="ag-theme-alpine",
                            ),
                        ],
                        style={"marginTop": "16px"},
                    ),
                ],
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 强化模型页相关回调。"""

    @app.callback(
        Output("model-sector-table", "rowData"),
        Input("rl-agent-store", "data"),
    )
    def render_model_parameters(agent_data):  # noqa: D401
        return core.render_model_parameters_logic(agent_data)

    @app.callback(
        Output("rl-model-table", "rowData"),
        Input("rl-agent-store", "data"),
    )
    def render_model_details(agent_data):  # noqa: D401
        return core.render_model_details_logic(agent_data)

    @app.callback(
        Output("rl-parameter-history", "data"),
        Output("log-store", "data", allow_duplicate=True),
        Input("rl-agent-store", "data"),
        State("rl-parameter-history", "data"),
        State("log-store", "data"),
        prevent_initial_call=True,
    )
    def update_parameter_history(agent_data, history_state, log_state):  # noqa: D401
        return core.update_parameter_history_logic(agent_data, history_state, log_state)

    @app.callback(
        Output("model-global-history-table", "rowData"),
        Output("model-sector-history-table", "rowData"),
        Input("rl-parameter-history", "data"),
    )
    def render_parameter_history_tables(history_state):  # noqa: D401
        return core.render_parameter_history_tables_logic(history_state)
