"""Root layout assembly and global log callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

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
            dcc.Interval(id="connection-poller", interval=60000, n_intervals=0),
            dcc.Interval(id="auto-run-trigger", interval=1500, n_intervals=0, max_intervals=1),
            dcc.Interval(id="post-open-eval", interval=300000, n_intervals=0),
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
                    dcc.Tab(label="数据连接", children=[build_connections_tab(config)]),
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
                    dbc.ModalFooter(dbc.Button("关闭", id="close-log-btn", color="secondary")),
                ],
                id="log-modal",
                is_open=False,
                size="lg",
                scrollable=True,
            ),
        ]
    )


def register_callbacks(app: Dash) -> None:
    """Bind global log viewers and modal toggles."""

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
