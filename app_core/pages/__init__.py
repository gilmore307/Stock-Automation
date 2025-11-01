"""Callback registration for each页面模块."""

from __future__ import annotations

from dash import Dash

from . import connections, earnings, logs, overview_tab, predictions, reinforcement, tasks, validation

__all__ = [
    "register_all_callbacks",
]


def register_all_callbacks(app: Dash) -> None:
    """Register callbacks for every页面."""

    connections.register_callbacks(app)
    earnings.register_callbacks(app)
    predictions.register_callbacks(app)
    validation.register_callbacks(app)
    overview_tab.register_callbacks(app)
    reinforcement.register_callbacks(app)
    tasks.register_callbacks(app)
    logs.register_callbacks(app)
