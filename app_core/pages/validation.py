"""预测验证回顾页的回调封装。"""

from __future__ import annotations

from dash import Dash, Input, Output, State

from .. import core


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
