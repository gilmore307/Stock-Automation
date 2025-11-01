"""Application helpers for the 财报博弈 Dash 仪表盘。"""

from __future__ import annotations

from .application import create_dash_app
from .pages.a_root import LayoutConfig, build_layout

__all__ = ["LayoutConfig", "build_layout", "create_dash_app"]
