"""财报日程页面与后台抓取的 Dash 回调封装。"""

from __future__ import annotations

from dash import Dash, Input, Output, State

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register 财报日程页相关回调。"""

    @app.callback(
        Output("run-id-store", "data"),
        Output("log-store", "data", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Output("table", "rowData", allow_duplicate=True),
        Output("log-poller", "disabled", allow_duplicate=True),
        Output("task-store", "data", allow_duplicate=True),
        Input("auto-run-trigger", "n_intervals"),
        Input("earnings-date-picker", "date"),
        Input("earnings-refresh-btn", "n_clicks"),
        Input("ft-session-store", "data"),
        State("ft-username", "value"),
        State("ft-password", "value"),
        State("ft-2fa", "value"),
        State("task-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def start_run(auto_intervals, selected_date, refresh_clicks, session_data, username, password, twofa, task_state):  # noqa: D401
        return core.start_run_logic(
            auto_intervals,
            selected_date,
            refresh_clicks,
            session_data,
            username,
            password,
            twofa,
            task_state,
        )

    @app.callback(
        Output("log-store", "data", allow_duplicate=True),
        Output("table", "rowData", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Output("ft-session-store", "data", allow_duplicate=True),
        Output("log-poller", "disabled", allow_duplicate=True),
        Output("task-store", "data", allow_duplicate=True),
        Input("log-poller", "n_intervals"),
        State("run-id-store", "data"),
        State("log-store", "data"),
        State("table", "rowData"),
        State("task-store", "data"),
        prevent_initial_call=True,
    )
    def poll_run_state(n_intervals, run_id, existing_logs, current_rows, task_state):  # noqa: D401
        del n_intervals
        return core.poll_run_state_logic(run_id, existing_logs, current_rows, task_state)

    @app.callback(
        Output("dl", "data"),
        Input("dl-btn", "n_clicks"),
        State("table", "rowData"),
        prevent_initial_call=True,
    )
    def download_csv(n, data):
        return core.download_csv_logic(n, data)
