"""Root layout assembly and global log callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from .. import core
from .b_connections import build_layout as build_connections_tab
from .c_overview import build_layout as build_overview_tab
from .d_earnings import build_layout as build_earnings_tab
from .e_tasks import build_layout as build_tasks_tab
from .f_predictions import build_layout as build_predictions_tab
from .g_validation import build_layout as build_validation_tab
from .h_reinforcement import build_layout as build_reinforcement_tab


@dataclass(slots=True)
class LayoutConfig:
    """Configuration container shared across page layouts."""

    default_username: str
    default_password: str
    default_picker_date: Any
    min_picker_date: Any
    max_picker_date: Any
    default_picker_date_str: str
    rl_snapshot: Any
    app_title: str
    navbar_title: str
    main_heading: str
    prediction_timelines: Sequence[dict[str, Any]]


def build_login_panel(config: LayoutConfig) -> html.Div:
    """构建在解锁主界面前展示的 Firstrade 登录面板。"""

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


def build_layout(config: LayoutConfig) -> html.Div:
    """Compose the full Dash layout tree."""

    heading_children: list[html.H2] = []
    if config.main_heading:
        heading_children.append(html.H2(config.main_heading))

    return html.Div(
        [
            dcc.Store(id="ft-session-store", storage_type="memory"),
            dcc.Store(id="log-store", storage_type="memory", data=[]),
            dcc.Store(id="log-autoscroll", storage_type="memory"),
            dcc.Store(id="task-store", storage_type="memory", data={"tasks": []}),
            dcc.Store(
                id="prediction-store",
                storage_type="memory",
                data={
                    "results": [],
                    "missing": [],
                    "errors": [],
                    "rl_snapshot": config.rl_snapshot,
                },
            ),
            dcc.Store(id="prediction-run-id-store", storage_type="memory"),
            dcc.Store(id="rl-agent-store", storage_type="memory", data=config.rl_snapshot),
            dcc.Store(
                id="rl-parameter-history",
                storage_type="memory",
                data={"global": {"last": {}, "changes": []}, "sectors": {}},
            ),
            dcc.Store(id="evaluation-store", storage_type="memory", data={}),
            dcc.Interval(id="prediction-poller", interval=1000, disabled=True),
            dcc.Interval(
                id="connection-poller", interval=60000, n_intervals=0, disabled=True
            ),
            dcc.Interval(
                id="auto-run-trigger",
                interval=1500,
                n_intervals=0,
                max_intervals=1,
                disabled=True,
            ),
            dcc.Interval(
                id="post-open-eval",
                interval=300000,
                n_intervals=0,
                disabled=True,
            ),
            html.Div(
                build_login_panel(config),
                id="login-shell",
            ),
            html.Div(
                [
                    dbc.Navbar(
                        dbc.Container(
                            [
                                dbc.NavbarBrand(config.navbar_title),
                                dbc.Button(
                                    "查看日志",
                                    id="show-log-btn",
                                    color="primary",
                                    className="ms-auto",
                                ),
                            ],
                            fluid=True,
                        ),
                        color="dark",
                        dark=True,
                        className="mb-3",
                    ),
                    *heading_children,
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="数据连接", children=[build_connections_tab(config)]
                            ),
                            dcc.Tab(label="预测概览", children=[build_overview_tab(config)]),
                            dcc.Tab(label="财报日程", children=[build_earnings_tab(config)]),
                            dcc.Tab(label="任务中心", children=[build_tasks_tab(config)]),
                            dcc.Tab(label="预测明细", children=[build_predictions_tab(config)]),
                            dcc.Tab(label="验证回顾", children=[build_validation_tab(config)]),
                            dcc.Tab(label="强化模型", children=[build_reinforcement_tab(config)]),
                        ]
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("执行日志")),
                            dbc.ModalBody(
                                html.Pre(
                                    id="log-output",
                                    style={"maxHeight": "60vh", "overflowY": "auto"},
                                )
                            ),
                            dbc.ModalFooter(
                                dbc.Button("关闭", id="close-log-btn", color="secondary")
                            ),
                        ],
                        id="log-modal",
                        is_open=False,
                        size="lg",
                        scrollable=True,
                    ),
                ],
                id="app-shell",
                style={"display": "none"},
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Bind global log viewers and modal toggles."""

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
    def run_login(n_clicks, username, password, twofa, log_state):
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
        Output("login-shell", "style"),
        Output("app-shell", "style"),
        Output("auto-run-trigger", "disabled"),
        Output("prediction-poller", "disabled"),
        Output("connection-poller", "disabled"),
        Output("post-open-eval", "disabled"),
        Input("ft-session-store", "data"),
    )
    def _toggle_gate(session_state):
        logged_in = bool(session_state and isinstance(session_state, dict) and session_state.get("sid"))
        if logged_in:
            login_style = {"display": "none"}
            app_style = {}
        else:
            login_style = {"padding": "40px 0"}
            app_style = {"display": "none"}
        disable_intervals = not logged_in
        return (
            login_style,
            app_style,
            disable_intervals,
            disable_intervals,
            disable_intervals,
            disable_intervals,
        )

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
