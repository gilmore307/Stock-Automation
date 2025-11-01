"""强化学习模型页面回调封装。"""

from __future__ import annotations

from dash import Dash, Input, Output, State

from .. import core


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
