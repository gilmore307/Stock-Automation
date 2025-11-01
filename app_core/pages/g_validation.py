"""预测验证回顾页的布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:
    """Return the validation tab layout."""

    return html.Div(
        [
            html.Div(
                "此页聚焦已完成检验的标的，可对比不同预测时点与实际果。",
                style={"margin": "8px 0", "whiteSpace": "pre-wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("选择决策日"),
                            dcc.Dropdown(id="validation-date-dropdown", options=[], value=None, clearable=False),
                        ],
                        style={"flex": 1, "minWidth": 220, "paddingRight": 12},
                    ),
                    html.Div(
                        [
                            html.Label("选择标的"),
                            dcc.Dropdown(id="validation-symbol-dropdown", options=[], value=None, clearable=False),
                        ],
                        style={"flex": 1, "minWidth": 220, "paddingRight": 12},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "12px"},
            ),
            html.Div(id="validation-status", style={"margin": "8px 0", "whiteSpace": "pre-wrap"}),
            dag.AgGrid(
                id="validation-table",
                columnDefs=[
                    {"headerName": "参数", "field": "parameter", "pinned": "left"},
                    *[
                        {
                            "headerName": cfg.get("label"),
                            "field": f"col_{cfg.get('key')}",
                            "minWidth": 140,
                        }
                        for cfg in config.prediction_timelines
                    ],
                    {"headerName": "实际结果", "field": "col_actual", "minWidth": 140},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": False,
                    "resizable": True,
                    "filter": False,
                    "flex": 1,
                    "wrapText": True,
                    "autoHeight": True,
                },
                dashGridOptions={"domLayout": "autoHeight", "suppressMovableColumns": True},
                className="ag-theme-alpine",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="validation-graph-dci"), md=12, lg=6),
                    dbc.Col(dcc.Graph(id="validation-graph-prob"), md=12, lg=6),
                ],
                className="gy-4",
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register 验证页相关回调。"""

    @app.callback(
        Output("validation-date-dropdown", "options"),
        Output("validation-date-dropdown", "value"),
        Output("validation-symbol-dropdown", "options"),
        Output("validation-symbol-dropdown", "value"),
        Output("validation-status", "children"),
        Output("validation-table", "rowData"),
        Output("validation-graph-dci", "figure"),
        Output("validation-graph-prob", "figure"),
        Input("validation-date-dropdown", "value"),
        Input("validation-symbol-dropdown", "value"),
        State("evaluation-store", "data"),
        State("prediction-store", "data"),
    )
    def render_validation_view(selected_date, selected_symbol, evaluation_store, prediction_store):  # noqa: D401
        return core.render_validation_view_logic(selected_date, selected_symbol, evaluation_store, prediction_store)
