"""预测相关页面的布局与回调封装。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_ag_grid as dag
from dash import Dash, Input, Output, State, html

from .. import core

if TYPE_CHECKING:
    from .a_root import LayoutConfig


def build_layout(config: "LayoutConfig") -> html.Div:  # noqa: D401
    del config
    return html.Div(
        [
            html.Div(
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


def register_callbacks(app: Dash) -> None:
    """Register 预测页相关回调。"""

    @app.callback(
        Output("prediction-store", "data"),
        Output("prediction-status", "children"),
        Output("rl-agent-store", "data"),
        Output("task-store", "data", allow_duplicate=True),
        Output("log-store", "data", allow_duplicate=True),
        Output("prediction-run-id-store", "data"),
        Output("prediction-poller", "disabled", allow_duplicate=True),
        Input("table", "selectedRows"),
        Input("table", "rowData"),
        Input("ft-session-store", "data"),
        State("task-store", "data"),
        State("earnings-date-picker", "date"),
        State("log-store", "data"),
        State("ft-username", "value"),
        State("ft-password", "value"),
        State("ft-2fa", "value"),
        State("prediction-run-id-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_predictions(
        selected_rows,
        row_data,
        session_state,
        task_state,
        picker_date,
        log_state,
        username,
        password,
        twofa,
        existing_run_id,
    ):  # noqa: D401
        return core.update_predictions_logic(
            selected_rows,
            row_data,
            session_state,
            task_state,
            picker_date,
            log_state,
            username,
            password,
            twofa,
            existing_run_id,
        )

    @app.callback(
        Output("prediction-table", "rowData"),
        Input("prediction-store", "data"),
    )
    def render_prediction_table(store_data):  # noqa: D401
        return core.render_prediction_table_logic(store_data)

    @app.callback(
        Output("prediction-store", "data", allow_duplicate=True),
        Output("prediction-status", "children", allow_duplicate=True),
        Output("rl-agent-store", "data", allow_duplicate=True),
        Output("task-store", "data", allow_duplicate=True),
        Output("log-store", "data", allow_duplicate=True),
        Output("prediction-poller", "disabled", allow_duplicate=True),
        Output("prediction-run-id-store", "data", allow_duplicate=True),
        Input("prediction-poller", "n_intervals"),
        State("prediction-run-id-store", "data"),
        State("prediction-store", "data"),
        State("prediction-status", "children"),
        State("rl-agent-store", "data"),
        State("task-store", "data"),
        State("log-store", "data"),
        prevent_initial_call=True,
    )
    def poll_prediction_run(
        n_intervals,
        run_id,
        existing_store,
        existing_status,
        existing_snapshot,
        existing_task_state,
        existing_logs,
    ):  # noqa: D401
        return core.poll_prediction_run_logic(
            n_intervals,
            run_id,
            existing_store,
            existing_status,
            existing_snapshot,
            existing_task_state,
            existing_logs,
        )

    @app.callback(
        Output("evaluation-store", "data"),
        Input("post-open-eval", "n_intervals"),
        State("evaluation-store", "data"),
    )
    def auto_evaluate_predictions(n_intervals, existing_store):  # noqa: D401
        return core.auto_evaluate_predictions_logic(n_intervals, existing_store)

    @app.callback(
        Output("prediction-store", "data", allow_duplicate=True),
        Input("evaluation-store", "data"),
        State("prediction-store", "data"),
        prevent_initial_call=True,
    )
    def sync_actual_into_predictions(evaluation_store, prediction_store):  # noqa: D401
        return core.sync_actual_into_predictions_logic(evaluation_store, prediction_store)
