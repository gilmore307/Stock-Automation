"""Firstrade 登录页组件与回调。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_login_panel(config: "LayoutConfig") -> html.Div:
    """构建 Firstrade 登录面板。"""

    form_controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Firstrade 用户名"),
                    dcc.Input(
                        id="ft-username",
                        type="text",
                        value=config.default_username,
                        debounce=True,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("密码"),
                    dcc.Input(
                        id="ft-password",
                        type="password",
                        value=config.default_password,
                        debounce=False,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("双重验证码（可选）"),
                    dcc.Input(
                        id="ft-2fa",
                        type="text",
                        value="",
                        debounce=False,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            dbc.Button("登录 Firstrade", id="login-btn", color="primary", className="w-100"),
        ]
    )

    help_block = html.Div(
        [
            html.P(
                "首次启动时请先完成 Firstrade 登录，系统会在会话建立后自动打开其他功能页面。",
                style={"marginBottom": "6px"},
            ),
            html.P(
                "若登录失败，可查看下方提示或打开日志弹窗以获取详细错误原因。",
                style={"marginBottom": "0"},
            ),
        ],
        style={"fontSize": "0.9rem", "color": "#555", "marginTop": "16px"},
    )

    return html.Div(
        [
            dbc.Container(
                [
                    html.H2("请登录 Firstrade", style={"textAlign": "center", "marginBottom": "24px"}),
                    dbc.Row(
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            form_controls,
                                            html.Div(
                                                id="login-status",
                                                style={
                                                    "marginTop": "16px",
                                                    "whiteSpace": "pre-wrap",
                                                    "minHeight": "24px",
                                                },
                                            ),
                                            help_block,
                                        ]
                                    )
                                ],
                                className="shadow-sm",
                            ),
                            width=12,
                            lg=6,
                            className="mx-auto",
                        )
                    ),
                ],
                fluid=True,
                className="py-5",
            )
        ],
        style={"minHeight": "100vh", "backgroundColor": "#f8f9fa"},
    )


def register_callbacks(app: Dash) -> None:
    """注册 Firstrade 登录按钮的回调。"""

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
