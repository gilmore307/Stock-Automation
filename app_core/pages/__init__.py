"""Callback registration for each 页面模块。"""

from __future__ import annotations

from dash import Dash

from . import (
    a_root,
    b_connections,
    c_overview,
    d_earnings,
    e_tasks,
    f_predictions,
    g_validation,
    h_reinforcement,
    login,
)

__all__ = ["register_all_callbacks"]


def register_all_callbacks(app: Dash) -> None:
    """Register callbacks for every 页面模块。"""

    login.register_callbacks(app)
    a_root.register_callbacks(app)
    b_connections.register_callbacks(app)
    c_overview.register_callbacks(app)
    d_earnings.register_callbacks(app)
    e_tasks.register_callbacks(app)
    f_predictions.register_callbacks(app)
    g_validation.register_callbacks(app)
    h_reinforcement.register_callbacks(app)
