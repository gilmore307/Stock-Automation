"""Dash 应用工厂与全局回调注册入口。"""

from __future__ import annotations

import os

import dash_bootstrap_components as dbc
from dash import Dash

from . import core
from .pages import register_all_callbacks
from .pages.a_root import LayoutConfig, build_layout

AG_GRID_STYLES = [
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css",
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine.css",
]


def create_dash_app() -> Dash:
    """构建并返回 Dash 应用实例。"""

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP] + AG_GRID_STYLES)
    app.title = "财报博弈自动化平台"

    default_picker_date, min_picker_date, max_picker_date = core._date_picker_bounds()
    layout_config = LayoutConfig(
        default_username=os.getenv("FTD_USERNAME", "sunweicheng"),
        default_password=os.getenv("FTD_PASSWORD", "Swc_661199"),
        default_picker_date=default_picker_date,
        min_picker_date=min_picker_date,
        max_picker_date=max_picker_date,
        default_picker_date_str=default_picker_date.isoformat(),
        rl_snapshot=core.RL_MANAGER.snapshot() if core.RL_MANAGER else None,
        app_title=app.title,
        navbar_title="财报博弈自动化平台",
        main_heading="",
        prediction_timelines=core.PREDICTION_TIMELINES,
    )

    app.layout = build_layout(layout_config)
    register_all_callbacks(app)

    return app
