"""数据连接状态页面的布局与回调。"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:
    """Return the tab contents for the connection screen."""

    return html.Div(
        [
            html.Div(
                "请确认 Firstrade 登录成功后，再使用下方按钮检查当前的账号与数据连接状态。",
                style={"marginBottom": "12px", "whiteSpace": "pre-wrap"},
            ),
            html.Div(
                [
                    dbc.Button("检查连接", id="check-connections-btn", color="info", size="sm"),
                    html.Div(
                        "系统会每分钟自动巡检连接状态，您也可以手动点击按钮立即更新。",
                        id="connection-status-area",
                        style={"marginTop": "10px", "whiteSpace": "pre-wrap"},
                    ),
                ],
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Register callbacks for连接巡检页面."""

    @app.callback(
        Output("connection-status-area", "children"),
        Input("check-connections-btn", "n_clicks"),
        Input("connection-poller", "n_intervals"),
        State("ft-session-store", "data"),
    )
    def refresh_connection_status(n_clicks, poll_intervals, ft_session):  # noqa: D401
        del n_clicks, poll_intervals
        session_data = ft_session if isinstance(ft_session, dict) else None
        statuses = core._check_resource_connections(session_data)  # noqa: SLF001
        checked_at = dt.datetime.now(core.US_EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        return core._render_connection_statuses(statuses, checked_at)  # noqa: SLF001
