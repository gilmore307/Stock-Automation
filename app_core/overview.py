"""预测概览页图表数据生成工具。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(t=40, l=40, r=40, b=40))
    return fig


def build_overview_figures(archive: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    """根据预测归档构建整体命中率、行业命中率和时间序列趋势图。"""

    if not isinstance(archive, dict) or not archive:
        msg = "暂无历史检验数据。"
        return (_empty_figure(msg), _empty_figure(msg), _empty_figure(msg))

    item_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for date_key, entry in archive.items():
        if not isinstance(entry, dict):
            continue
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        summary = evaluation.get("summary")
        if isinstance(summary, dict):
            summary_rows.append(
                {
                    "date": summary.get("date") or date_key,
                    "success_rate": float(summary.get("success_rate") or 0.0),
                    "total": summary.get("total") or 0,
                    "correct": summary.get("correct") or 0,
                }
            )
        items = evaluation.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            timeline_key = str(item.get("timeline_key") or "")
            if timeline_key and timeline_key != "decision_day":
                continue
            row = {
                "date": date_key,
                "sector": str(item.get("sector") or "未知"),
                "prediction_correct": str(item.get("prediction_correct") or ""),
            }
            item_rows.append(row)

    if not item_rows:
        msg = "尚未生成可用于统计的预测检验记录。"
        return (_empty_figure(msg), _empty_figure(msg), _empty_figure(msg))

    items_df = pd.DataFrame(item_rows)
    items_df["is_correct"] = items_df["prediction_correct"].astype(str) == "是"

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("date")

    counts = items_df["is_correct"].value_counts().to_dict()
    success = int(counts.get(True, 0))
    failure = int(counts.get(False, 0))
    total_fig = px.bar(
        x=["命中", "未命中"],
        y=[success, failure],
        title="整体命中 vs. 失误次数",
        labels={"x": "结果", "y": "次数"},
        text=[success, failure],
    )
    total_fig.update_layout(yaxis_title="次数", xaxis_title="结果")

    sector_df = items_df.groupby("sector")["is_correct"].mean().reset_index()
    sector_df["success_pct"] = sector_df["is_correct"] * 100
    sector_fig = px.bar(
        sector_df,
        x="sector",
        y="success_pct",
        title="各行业命中率",
        labels={"sector": "行业", "success_pct": "命中率(%)"},
        text=sector_df["success_pct"].round(1),
    )
    sector_fig.update_layout(yaxis=dict(range=[0, 100]))

    if summary_df.empty:
        trend_df = (
            items_df.groupby("date")["is_correct"].mean().reset_index().sort_values("date")
        )
        trend_df["success_rate"] = trend_df["is_correct"] * 100
    else:
        trend_df = summary_df.copy()
        trend_df["success_rate"] = trend_df["success_rate"].astype(float) * 100

    trend_fig = px.line(
        trend_df,
        x="date",
        y="success_rate",
        markers=True,
        title="时间序列命中率",
        labels={"date": "日期", "success_rate": "命中率(%)"},
    )
    trend_fig.update_layout(yaxis=dict(range=[0, 100]))

    return total_fig, sector_fig, trend_fig
