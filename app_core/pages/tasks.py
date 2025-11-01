"""任务中心相关回调。"""

from __future__ import annotations

from dash import Dash, Input, Output

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register 任务中心的渲染回调。"""

    @app.callback(
        Output("task-table", "rowData"),
        Input("task-store", "data"),
    )
    def render_task_table(task_state):  # noqa: D401
        return core.render_task_table_logic(task_state)
