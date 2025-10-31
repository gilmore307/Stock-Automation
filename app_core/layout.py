"""Dash 布局模块，负责生成各个页签的布局组件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html


@dataclass(slots=True)
class LayoutConfig:
    """构建布局所需的上下文配置。"""

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


def _build_connection_tab(config: LayoutConfig) -> html.Div:
    return html.Div(
        [
            html.Div(
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
                        style={"flex": 1, "minWidth": 240, "paddingRight": 10},
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
                        style={"flex": 1, "minWidth": 240, "paddingRight": 10},
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
                        style={"flex": 1, "minWidth": 200, "paddingRight": 10},
                    ),
                    html.Div(
                        [
                            html.Label("操作"),
                            html.Button("登录", id="login-btn"),
                        ],
                        style={"display": "flex", "flexDirection": "column", "justifyContent": "flex-end"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "marginBottom": "10px"},
            ),
            html.Div(id="login-status", style={"margin": "4px 0", "whiteSpace": "pre-wrap"}),
            html.Div(
                [
                    dbc.Button("检查连接", id="check-connections-btn", color="info", size="sm"),
                    html.Div(
                        "系统会每分钟自动巡检连接状态，您也可以手动点击按钮立即更新。",
                        id="connection-status-area",
                        style={"marginTop": "10px", "whiteSpace": "pre-wrap"},
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ]
    )


def _build_overview_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                "该页面展示整体预测命中率以及按行业板块拆分的表现，帮助快速把握模型健康度。",
                style={"margin": "8px 0", "whiteSpace": "pre-wrap"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="overview-total-success"),
                        lg=6,
                        md=12,
                    ),
                    dbc.Col(
                        dcc.Graph(id="overview-sector-success"),
                        lg=6,
                        md=12,
                    ),
                ],
                className="gy-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="overview-timeline-trend"),
                        lg=12,
                        md=12,
                    ),
                ],
                className="gy-4",
            ),
        ]
    )


def _build_earnings_tab(config: LayoutConfig) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("选择决策日"),
                    dcc.DatePickerSingle(
                        id="earnings-date-picker",
                        date=config.default_picker_date_str,
                        min_date_allowed=config.min_picker_date,
                        max_date_allowed=config.max_picker_date,
                        display_format="YYYY-MM-DD",
                        initial_visible_month=config.default_picker_date,
                        day_size=36,
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Button("下载 CSV", id="dl-btn"),
                ],
                style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
            ),
            html.Div(id="status", style={"margin": "8px 0", "whiteSpace": "pre-wrap"}),
            dag.AgGrid(
                id="table",
                columnDefs=[
                    {"headerName": "代码", "field": "symbol", "filter": "agTextColumnFilter"},
                    {"headerName": "公司", "field": "company", "filter": "agTextColumnFilter"},
                    {"headerName": "决策日期", "field": "decision_date", "filter": "agDateColumnFilter"},
                    {"headerName": "时间段", "field": "bucket", "filter": "agTextColumnFilter"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "flex": 1,
                    "minWidth": 120,
                },
                dashGridOptions={
                    "domLayout": "autoHeight",
                    "pagination": True,
                    "paginationPageSize": 20,
                    "animateRows": True,
                    "rowSelection": "multiple",
                },
                style={"height": "600px", "width": "100%"},
                className="ag-theme-alpine",
            ),
            dcc.Download(id="dl"),
        ]
    )


def _build_task_tab() -> html.Div:
    return html.Div(
        [
            dag.AgGrid(
                id="task-table",
                columnDefs=[
                    {"headerName": "任务", "field": "name"},
                    {"headerName": "状态", "field": "status"},
                    {"headerName": "详情", "field": "detail"},
                    {"headerName": "开始时间", "field": "start_time"},
                    {"headerName": "结束时间", "field": "end_time"},
                    {"headerName": "更新时间", "field": "updated_at"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": False,
                    "resizable": True,
                    "filter": False,
                    "flex": 1,
                    "minWidth": 140,
                },
                dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                style={"width": "100%"},
                className="ag-theme-alpine",
            ),
        ]
    )


def _build_prediction_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                "系统会根据财报列表自动生成 DCI 预测；此处仅展示尚未出结果的标的。",
                id="prediction-status",
                style={"margin": "8px 0", "whiteSpace": "pre-wrap"},
            ),
            dag.AgGrid(
                id="prediction-table",
                columnDefs=[
                    {"headerName": "代码", "field": "symbol", "filter": "agTextColumnFilter"},
                    {"headerName": "公司", "field": "company", "filter": "agTextColumnFilter"},
                    {"headerName": "行业板块", "field": "sector", "filter": "agTextColumnFilter"},
                    {"headerName": "决策日期", "field": "decision_date", "filter": "agDateColumnFilter"},
                    {"headerName": "时间段", "field": "bucket", "filter": "agTextColumnFilter"},
                    {"headerName": "预测时点", "field": "timeline_label"},
                    {"headerName": "距决策日(天)", "field": "lookback_days"},
                    {"headerName": "方向", "field": "direction"},
                    {"headerName": "上涨概率(%)", "field": "p_up_pct"},
                    {"headerName": "基础DCI", "field": "dci_base"},
                    {"headerName": "惩罚后DCI", "field": "dci_penalised"},
                    {"headerName": "最终DCI", "field": "dci_final"},
                    {"headerName": "确定性", "field": "certainty"},
                    {"headerName": "仓位权重", "field": "position_weight"},
                    {"headerName": "仓位建议", "field": "position_bucket"},
                    {"headerName": "基础方向分(S)", "field": "base_score"},
                    {"headerName": "一致性收缩", "field": "shrink_eg"},
                    {"headerName": "拥挤收缩", "field": "shrink_ci"},
                    {"headerName": "分歧收缩", "field": "shrink_disagreement"},
                    {"headerName": "宏观收缩", "field": "shrink_shock"},
                    {"headerName": "一致性z", "field": "input_z_cons"},
                    {"headerName": "叙事z", "field": "input_z_narr"},
                    {"headerName": "拥挤度", "field": "input_ci"},
                    {"headerName": "质量分", "field": "input_q"},
                    {"headerName": "分歧度", "field": "input_d"},
                    {"headerName": "预期波动(%)", "field": "input_em_pct"},
                    {"headerName": "稳定性", "field": "input_s_stab"},
                    {"headerName": "震荡日标记", "field": "input_shock_flag"},
                    {"headerName": "强化学习方向", "field": "rl_direction"},
                    {"headerName": "强化学习上涨概率(%)", "field": "rl_p_up_pct"},
                    {"headerName": "概率调整(百分点)", "field": "rl_delta_pct"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "flex": 1,
                    "minWidth": 120,
                },
                dashGridOptions={
                    "domLayout": "autoHeight",
                    "pagination": True,
                    "paginationPageSize": 15,
                    "animateRows": True,
                },
                style={"height": "500px", "width": "100%"},
                className="ag-theme-alpine",
            ),
        ]
    )


def _build_validation_tab(config: LayoutConfig) -> html.Div:
    return html.Div(
        [
            html.Div(
                "此页聚焦已完成检验的标的，可对比不同预测时点与实际结果。",
                style={"margin": "8px 0", "whiteSpace": "pre-wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("选择决策日"),
                            dcc.Dropdown(id="validation-date-dropdown", options=[], value=None, clearable=False),
                        ],
                        style={"flex": 1, "minWidth": 220, "paddingRight": 12},
                    ),
                    html.Div(
                        [
                            html.Label("选择标的"),
                            dcc.Dropdown(id="validation-symbol-dropdown", options=[], value=None, clearable=False),
                        ],
                        style={"flex": 1, "minWidth": 220, "paddingRight": 12},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "12px"},
            ),
            html.Div(id="validation-status", style={"margin": "8px 0", "whiteSpace": "pre-wrap"}),
            dag.AgGrid(
                id="validation-table",
                columnDefs=[
                    {"headerName": "参数", "field": "parameter", "pinned": "left"},
                    *[
                        {
                            "headerName": cfg.get("label"),
                            "field": f"col_{cfg.get('key')}",
                            "minWidth": 140,
                        }
                        for cfg in config.prediction_timelines
                    ],
                    {"headerName": "实际结果", "field": "col_actual", "minWidth": 140},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": False,
                    "resizable": True,
                    "filter": False,
                    "flex": 1,
                    "wrapText": True,
                    "autoHeight": True,
                },
                dashGridOptions={"domLayout": "autoHeight", "suppressMovableColumns": True},
                className="ag-theme-alpine",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="validation-graph-dci"), md=12, lg=6),
                    dbc.Col(dcc.Graph(id="validation-graph-prob"), md=12, lg=6),
                ],
                className="gy-4",
            ),
        ]
    )


def _build_model_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                "本页展示强化学习模型的全局与行业级参数，所有更新均由自动检验驱动。",
                style={"margin": "8px 0", "whiteSpace": "pre-wrap"},
            ),
            html.Div(
                [
                    html.H5("全局模型概览"),
                    html.Pre(
                        id="model-global-summary",
                        style={
                            "maxHeight": "240px",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                            "backgroundColor": "#f8f9fa",
                            "padding": "12px",
                            "borderRadius": "6px",
                        },
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.H5("行业模型概览"),
                    dag.AgGrid(
                        id="model-sector-table",
                        columnDefs=[
                            {"headerName": "行业", "field": "sector"},
                            {"headerName": "预测次数", "field": "total_predictions"},
                            {"headerName": "更新次数", "field": "update_count"},
                            {"headerName": "当前基准", "field": "baseline"},
                            {"headerName": "主要权重", "field": "top_weights"},
                        ],
                        rowData=[],
                        defaultColDef={
                            "sortable": True,
                            "resizable": True,
                            "filter": True,
                            "flex": 1,
                            "minWidth": 140,
                        },
                        dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                        className="ag-theme-alpine",
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Hr(),
            dag.AgGrid(
                id="rl-model-table",
                columnDefs=[
                    {"headerName": "模型", "field": "model"},
                    {"headerName": "参数", "field": "parameter"},
                    {"headerName": "取值", "field": "value"},
                    {"headerName": "用途说明", "field": "description"},
                ],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "flex": 1,
                    "minWidth": 140,
                },
                dashGridOptions={"domLayout": "autoHeight", "pagination": False},
                className="ag-theme-alpine",
            ),
        ]
    )


def build_layout(config: LayoutConfig) -> html.Div:
    heading_children = []
    if config.main_heading:
        heading_children.append(html.H2(config.main_heading))

    return html.Div(
        [
            dcc.Store(id="ft-session-store", storage_type="memory"),
            dcc.Store(id="log-store", storage_type="memory", data=[]),
            dcc.Store(id="log-autoscroll", storage_type="memory"),
            dcc.Store(id="run-id-store", storage_type="memory"),
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
            dcc.Store(id="rl-agent-store", storage_type="memory", data=config.rl_snapshot),
            dcc.Store(id="evaluation-store", storage_type="memory", data={}),
            dcc.Interval(id="log-poller", interval=1000, disabled=True),
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
                    dcc.Tab(label="数据连接", children=[_build_connection_tab(config)]),
                    dcc.Tab(label="预测概览", children=[_build_overview_tab()]),
                    dcc.Tab(label="财报日程", children=[_build_earnings_tab(config)]),
                    dcc.Tab(label="任务中心", children=[_build_task_tab()]),
                    dcc.Tab(label="预测明细", children=[_build_prediction_tab()]),
                    dcc.Tab(label="验证回顾", children=[_build_validation_tab(config)]),
                    dcc.Tab(label="强化模型", children=[_build_model_tab()]),
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
