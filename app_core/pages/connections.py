"""Firstrade 登录与数据连接页面的回调逻辑。"""

from __future__ import annotations

import datetime as dt

from dash import Dash, Input, Output, State, no_update

from .. import core


def register_callbacks(app: Dash) -> None:
    """Register callbacks for连接与巡检页面."""

    @app.callback(
        Output("ft-session-store", "data"),
        Output("login-status", "children"),
        Output("log-store", "data", allow_duplicate=True),
        Input("login-btn", "n_clicks"),
        State("ft-username", "value"),
        State("ft-password", "value"),
        State("ft-2fa", "value"),
        State("log-store", "data"),
        prevent_initial_call=True,
    )
    def run_login(n_clicks, username, password, twofa, log_state):  # noqa: D401
        del n_clicks
        username = (username or "").strip()
        password = password or ""
        twofa = (twofa or "").strip()
        logs = core.append_log(
            log_state,
            f"收到用户“{username or '（空）'}”的登录请求。",
            task_label="登录",
        )

        def log(message: str) -> None:
            nonlocal logs
            logs = core.append_log(logs, message, task_label="登录")

        if not username or not password:
            log("登录终止：必须填写用户名和密码。")
            return no_update, "Firstrade 登录失败：请填写用户名和密码。", logs

        log("正在尝试登录 Firstrade……")
        ft = core.FTClient(
            username=username,
            password=password,
            twofa_code=twofa if twofa else None,
            logger=log,
        )
        if ft.enabled:
            state = ft.export_session_state()
            msg = f"Firstrade 登录成功：会话 {state.get('sid', '')[:4]}..."
            log("Firstrade 登录成功。")
            return state, msg, logs

        log(f"Firstrade 登录失败：{ft.error or '未知错误'}。")
        return {}, f"Firstrade 登录失败：{ft.error or '未知错误'}", logs

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
