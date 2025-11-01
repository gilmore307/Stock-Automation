"""全局日志组件相关回调。"""

from __future__ import annotations

from dash import Dash, Input, Output, State

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register 全局日志回调。"""

    @app.callback(
        Output("log-output", "children"),
        Input("log-store", "data"),
    )
    def update_log_output(log_state):
        return core.update_log_output_logic(log_state)

    @app.callback(
        Output("log-modal", "is_open"),
        Input("show-log-btn", "n_clicks"),
        Input("close-log-btn", "n_clicks"),
        State("log-modal", "is_open"),
    )
    def toggle_log_modal(show_clicks, close_clicks, is_open):
        return core.toggle_log_modal_logic(show_clicks, close_clicks, is_open)


    app.clientside_callback(  # type: ignore[misc]
        """
        function(children) {
            const el = document.getElementById('log-output');
            if (el) {
                requestAnimationFrame(() => {
                    el.scrollTop = el.scrollHeight;
                });
            }
            return null;
        }
        """,
        Output("log-autoscroll", "data"),
        Input("log-output", "children"),
        prevent_initial_call=True,
    )
