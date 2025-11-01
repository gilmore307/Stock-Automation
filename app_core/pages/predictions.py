"""预测相关页面的回调封装。"""

from __future__ import annotations

from dash import Dash, Input, Output, State

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register 预测页相关回调。"""

    @app.callback(
        Output("prediction-store", "data"),
        Output("prediction-status", "children"),
        Output("rl-agent-store", "data"),
        Output("task-store", "data", allow_duplicate=True),
        Output("log-store", "data", allow_duplicate=True),
        Input("table", "selectedRows"),
        Input("table", "rowData"),
        Input("ft-session-store", "data"),
        State("task-store", "data"),
        State("earnings-date-picker", "date"),
        State("log-store", "data"),
        State("ft-username", "value"),
        State("ft-password", "value"),
        State("ft-2fa", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def update_predictions(selected_rows, row_data, session_state, task_state, picker_date, log_state, username, password, twofa):  # noqa: D401
        return core.update_predictions_logic(
            selected_rows,
            row_data,
            session_state,
            task_state,
            picker_date,
            log_state,
            username,
            password,
            twofa,
        )

    @app.callback(
        Output("prediction-table", "rowData"),
        Input("prediction-store", "data"),
    )
    def render_prediction_table(store_data):  # noqa: D401
        return core.render_prediction_table_logic(store_data)

    @app.callback(
        Output("evaluation-store", "data"),
        Input("post-open-eval", "n_intervals"),
        State("evaluation-store", "data"),
    )
    def auto_evaluate_predictions(n_intervals, existing_store):  # noqa: D401
        return core.auto_evaluate_predictions_logic(n_intervals, existing_store)

    @app.callback(
        Output("prediction-store", "data", allow_duplicate=True),
        Input("evaluation-store", "data"),
        State("prediction-store", "data"),
        prevent_initial_call=True,
    )
    def sync_actual_into_predictions(evaluation_store, prediction_store):  # noqa: D401
        return core.sync_actual_into_predictions_logic(evaluation_store, prediction_store)
