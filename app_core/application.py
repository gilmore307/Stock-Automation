import os
import json
import time
import threading
import typing as T
import datetime as dt
import uuid
import copy
from pathlib import Path
from urllib.parse import urlparse

import requests
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from dci import BASE_FACTOR_WEIGHTS, build_inputs, compute_dci, get_factor_weights

try:
    from dci_rl import RLAgentManager, get_global_manager
except Exception:  # pragma: no cover - RL module is optional
    RL_MANAGER: "RLAgentManager | None" = None
else:
    RL_MANAGER = get_global_manager()

from .layout import LayoutConfig, build_layout
from .overview import build_overview_figures

# ---------- Timezone helpers ----------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

US_EASTERN = ZoneInfo("America/New_York")

RUN_LOCK = threading.Lock()
RUN_STATES: dict[str, dict[str, T.Any]] = {}
NO_UPDATE_SENTINEL = object()

ARCHIVE_ROOT = Path(__file__).with_name("archives")

PREDICTION_LOCK = threading.Lock()
PREDICTION_ARCHIVE_DIR = ARCHIVE_ROOT / "predictions"
PREDICTION_ARCHIVE_PATH = Path(__file__).with_name("prediction_archive.json")

SYMBOL_META_LOCK = threading.Lock()
SYMBOL_META_PATH = Path(__file__).with_name("symbol_metadata.json")
SYMBOL_META_CACHE: dict[str, dict[str, T.Any]] = {}

FINNHUB_API_KEY = os.getenv(
    "FINNHUB_API_KEY",
    "d3ifbshr01qn6oiodof0d3ifbshr01qn6oiodofg",
)
OPENFIGI_API_KEY = os.getenv(
    "OPENFIGI_API_KEY",
    "9e242491-ee71-47c0-9e04-49d2e952c15c",
)
FRED_API_KEY = os.getenv(
    "FRED_API_KEY",
    "5c9129e297742bb633b85e498edf83fa",
)

DEFAULT_DCI_DATA_PATH = Path(__file__).with_name("dci_inputs.json")
DCI_DATA_PATH = Path(os.getenv("DCI_DATA_PATH", str(DEFAULT_DCI_DATA_PATH)))

PREDICTION_TIMELINES = [
    {
        "key": "minus14",
        "label": "决策日前14天",
        "lookback": -14,
        "aliases": ["minus14", "-14", "t-14", "d-14"],
    },
    {
        "key": "minus7",
        "label": "决策日前7天",
        "lookback": -7,
        "aliases": ["minus7", "-7", "t-7", "d-7"],
    },
    {
        "key": "minus3",
        "label": "决策日前3天",
        "lookback": -3,
        "aliases": ["minus3", "-3", "t-3", "d-3"],
    },
    {
        "key": "minus1",
        "label": "决策日前1天",
        "lookback": -1,
        "aliases": ["minus1", "-1", "t-1", "d-1"],
    },
    {
        "key": "decision_day",
        "label": "决策日收盘前",
        "lookback": 0,
        "aliases": ["decision_day", "day0", "decision", "today", "0", "final"],
    },
]

EARNINGS_CACHE_PATH = Path(__file__).with_name("earnings_cache.json")
EARNINGS_ARCHIVE_DIR = ARCHIVE_ROOT / "earnings"
CACHE_LOCK = threading.Lock()

PARAMETER_ARCHIVE_DIR = ARCHIVE_ROOT / "parameter_changes"


def _load_archive_directory(directory: Path) -> dict[str, T.Any]:
    result: dict[str, T.Any] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        if not path.is_file():
            continue
        key = path.stem
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            result[key] = payload
    return result


def _write_archive_file(path: Path, payload: dict[str, T.Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _safe_timestamp_to_filename(timestamp: str) -> str:
    safe = timestamp.strip().replace(" ", "T").replace(":", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"-", "_", "T"})
    if not safe:
        safe = dt.datetime.now(US_EASTERN).strftime("%Y%m%dT%H%M%S")
    return safe


def _prepare_unique_archive_path(directory: Path, base_name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{base_name}.json"
    if not candidate.exists():
        return candidate
    suffix = uuid.uuid4().hex[:6]
    return directory / f"{base_name}-{suffix}.json"


def _earnings_archive_file(date_value: dt.date) -> Path:
    return EARNINGS_ARCHIVE_DIR / f"{date_value.isoformat()}.json"


def _prediction_archive_file(date_value: dt.date) -> Path:
    return PREDICTION_ARCHIVE_DIR / f"{date_value.isoformat()}.json"


def _persist_parameter_snapshot(timestamp: str, snapshot: dict[str, T.Any]) -> None:
    base_name = _safe_timestamp_to_filename(timestamp)
    path = _prepare_unique_archive_path(PARAMETER_ARCHIVE_DIR, base_name)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _describe_timeline_task(target_date: dt.date, timeline_cfg: dict[str, T.Any]) -> tuple[str, str, str]:
    key = str(timeline_cfg.get("key"))
    lookback = int(timeline_cfg.get("lookback", 0) or 0)
    offset_label = f"T{lookback:+d}"
    name = f"{target_date.strftime('%m月%d日')} 的 {offset_label} 天预测"
    return f"predict::{key}", name, offset_label

RL_PARAM_DESCRIPTIONS: dict[str, str] = {
    "learning_rate": "学习率：控制每次权重更新的幅度。",
    "gamma": "折扣因子：决定历史奖励对当前调整的影响。",
    "adjustment_scale": "调整幅度：限制概率校正的最大范围。",
    "bias": "方向偏置：对整体方向概率的固定修正。",
    "baseline": "基准：模型估计的平均命中率或预期收益。",
    "update_count": "更新次数：累计训练迭代的次数。",
    "total_predictions": "预测次数：该模型记录的总预测量。",
    "weights": "特征权重：表示对应信号对预测的影响强度。",
    "factor_weights": "DCI 因子权重：用于基础方向合成的权重分布。",
}

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "EPS_Rev_30d": "近30日每股收益一致预期修正。",
    "Sales_Rev_30d": "近30日营收一致预期修正。",
    "Guide_Drift": "公司指引偏移程度。",
    "Backlog_Bookings_Delta": "订单/积压的变化趋势。",
    "GM_YoY_Delta": "毛利率同比变化幅度。",
    "OPM_YoY_Delta": "营业利润率同比变化幅度。",
    "FCF_Margin_Slope": "自由现金流率的斜率表现。",
    "Ret20_rel": "近20日相对回报表现。",
    "Ret60_rel": "近60日相对回报表现。",
    "UDVol20": "20日上下行成交量比。",
    "Value_vs_Sector": "相对于行业的估值分位。",
    "EarningsYield_vs_Sector": "相对于行业的盈利收益率。",
}

TASK_ORDER = [f"predict::{cfg['key']}" for cfg in PREDICTION_TIMELINES]

VALIDATION_PARAMETER_SPECS: list[dict[str, T.Any]] = [
    {"label": "方向", "field": "direction", "fmt": "text", "actual_field": "actual_direction"},
    {"label": "上涨概率(%)", "field": "p_up_pct", "fmt": "float2"},
    {"label": "基础DCI", "field": "dci_base", "fmt": "float2"},
    {"label": "惩罚后DCI", "field": "dci_penalised", "fmt": "float2"},
    {"label": "最终DCI", "field": "dci_final", "fmt": "float2"},
    {"label": "确定性", "field": "certainty", "fmt": "float2"},
    {"label": "仓位权重", "field": "position_weight", "fmt": "float2"},
    {"label": "仓位建议", "field": "position_bucket", "fmt": "text"},
    {"label": "基础方向分(S)", "field": "base_score", "fmt": "float3"},
    {"label": "一致性收缩", "field": "shrink_factors.shrink_EG", "fmt": "float3"},
    {"label": "拥挤收缩", "field": "shrink_factors.shrink_CI", "fmt": "float3"},
    {"label": "分歧收缩", "field": "shrink_factors.disagreement", "fmt": "float3"},
    {"label": "宏观收缩", "field": "shrink_factors.shock", "fmt": "float3"},
    {"label": "一致性z", "field": "inputs.z_cons", "fmt": "float2"},
    {"label": "叙事z", "field": "inputs.z_narr", "fmt": "float2"},
    {"label": "拥挤度", "field": "inputs.CI", "fmt": "float1"},
    {"label": "质量分", "field": "inputs.Q", "fmt": "float1"},
    {"label": "分歧度", "field": "inputs.D", "fmt": "float2"},
    {"label": "预期波动(%)", "field": "inputs.EM_pct", "fmt": "float2"},
    {"label": "稳定性", "field": "inputs.S_stab", "fmt": "float2"},
    {"label": "震荡日标记", "field": "inputs.shock_flag", "fmt": "int"},
    {"label": "强化学习方向", "field": "rl_direction", "fmt": "text"},
    {"label": "强化学习上涨概率(%)", "field": "rl_p_up_pct", "fmt": "float2"},
    {"label": "概率调整(百分点)", "field": "rl_delta_pct", "fmt": "float2"},
]

for factor_name in BASE_FACTOR_WEIGHTS:
    VALIDATION_PARAMETER_SPECS.append(
        {
            "label": f"因子 {factor_name}",
            "field": f"scaled_factors.{factor_name}",
            "fmt": "float3",
        }
    )

for factor_name in BASE_FACTOR_WEIGHTS:
    VALIDATION_PARAMETER_SPECS.append(
        {
            "label": f"权重 {factor_name}",
            "field": f"factor_weights.{factor_name}",
            "fmt": "float3",
        }
    )

VALIDATION_PARAMETER_SPECS.append(
    {
        "label": "实际涨跌幅(%)",
        "field": None,
        "fmt": "text",
        "actual_field": "actual_move_pct",
        "actual_fmt": "float2",
    }
)


def _resolve_field(data: dict[str, T.Any] | None, field: str | None) -> T.Any:
    if not field or not data:
        return None
    parts = [part for part in field.split(".") if part]
    current: T.Any = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _format_value(value: T.Any, fmt: str | None) -> str:
    if value is None:
        return "-"
    if fmt == "text" or fmt is None:
        return str(value)
    if fmt == "int":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    if fmt.startswith("float"):
        precision_part = fmt.replace("float", "")
        try:
            precision = int(precision_part) if precision_part else 2
        except ValueError:
            precision = 2
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _build_validation_rows(
    timeline_map: dict[str, dict[str, T.Any]],
    actual_entry: dict[str, T.Any] | None,
) -> list[dict[str, T.Any]]:
    rows: list[dict[str, T.Any]] = []
    for spec in VALIDATION_PARAMETER_SPECS:
        row = {"parameter": spec.get("label")}
        for cfg in PREDICTION_TIMELINES:
            timeline_key = str(cfg.get("key"))
            entry = timeline_map.get(timeline_key)
            value = _resolve_field(entry, spec.get("field"))
            row[f"col_{timeline_key}"] = _format_value(value, spec.get("fmt"))
        actual_field = spec.get("actual_field")
        actual_value = _resolve_field(actual_entry, actual_field) if actual_field else None
        actual_fmt = spec.get("actual_fmt") or spec.get("fmt")
        row["col_actual"] = _format_value(actual_value, actual_fmt)
        rows.append(row)
    return rows


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        height=340,
    )
    return fig


def _build_validation_figures(
    timeline_map: dict[str, dict[str, T.Any]],
    actual_entry: dict[str, T.Any] | None,
) -> tuple[go.Figure, go.Figure]:
    labels: list[str] = []
    dci_final: list[float | None] = []
    dci_penalised: list[float | None] = []
    dci_base: list[float | None] = []
    prob_values: list[float | None] = []
    rl_prob_values: list[float | None] = []

    for cfg in PREDICTION_TIMELINES:
        key = str(cfg.get("key"))
        entry = timeline_map.get(key)
        if not isinstance(entry, dict):
            continue
        labels.append(str(cfg.get("label")))
        try:
            dci_final.append(float(entry.get("dci_final")))
        except (TypeError, ValueError):
            dci_final.append(None)
        try:
            dci_penalised.append(float(entry.get("dci_penalised")))
        except (TypeError, ValueError):
            dci_penalised.append(None)
        try:
            dci_base.append(float(entry.get("dci_base")))
        except (TypeError, ValueError):
            dci_base.append(None)
        try:
            prob_values.append(float(entry.get("p_up_pct")))
        except (TypeError, ValueError):
            prob_values.append(None)
        rl_val = entry.get("rl_p_up_pct")
        try:
            rl_prob_values.append(float(rl_val) if rl_val is not None else None)
        except (TypeError, ValueError):
            rl_prob_values.append(None)

    if not labels:
        return _empty_figure("暂无预测数据"), _empty_figure("暂无预测数据")

    fig_dci = go.Figure()
    fig_dci.add_trace(
        go.Scatter(x=labels, y=dci_base, mode="lines+markers", name="基础DCI")
    )
    fig_dci.add_trace(
        go.Scatter(x=labels, y=dci_penalised, mode="lines+markers", name="惩罚后DCI")
    )
    fig_dci.add_trace(
        go.Scatter(x=labels, y=dci_final, mode="lines+markers", name="最终DCI")
    )
    fig_dci.update_layout(
        title="不同时点的 DCI 变化",
        template="plotly_white",
        xaxis_title="预测时点",
        yaxis_title="DCI",
        height=360,
    )

    fig_prob = go.Figure()
    fig_prob.add_trace(
        go.Scatter(x=labels, y=prob_values, mode="lines+markers", name="上涨概率")
    )
    if any(value is not None for value in rl_prob_values):
        fig_prob.add_trace(
            go.Scatter(x=labels, y=rl_prob_values, mode="lines+markers", name="RL 概率")
        )
    fig_prob.update_layout(
        title="概率与强化学习调整对比",
        template="plotly_white",
        xaxis_title="预测时点",
        yaxis_title="概率(%)",
        height=360,
    )

    if isinstance(actual_entry, dict):
        move_pct = actual_entry.get("actual_move_pct")
        if move_pct is not None:
            try:
                move_value = float(move_pct)
            except (TypeError, ValueError):
                move_value = None
            if move_value is not None:
                fig_prob.add_hline(y=move_value, line_dash="dot", line_color="firebrick", name="实际涨跌幅")

    return fig_dci, fig_prob
TASK_TEMPLATES: dict[str, dict[str, T.Any]] = {}
for _timeline_cfg in PREDICTION_TIMELINES:
    _key = str(_timeline_cfg.get("key"))
    TASK_TEMPLATES[f"predict::{_key}"] = {
        "id": f"predict::{_key}",
        "name": f"{_timeline_cfg.get('label', _key)} 预测",
    }


def _coerce_non_negative_int(value: T.Any, fallback: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(fallback, 0)
    return max(parsed, 0)


def _ensure_task_progress(entry: dict[str, T.Any]) -> dict[str, T.Any]:
    total = _coerce_non_negative_int(entry.get("total_symbols"), 0)
    completed = _coerce_non_negative_int(entry.get("completed_symbols"), 0)
    processed = _coerce_non_negative_int(entry.get("processed_symbols"), completed)
    if processed < completed:
        processed = completed
    entry["total_symbols"] = total
    entry["completed_symbols"] = completed
    entry["processed_symbols"] = processed
    if total <= 0 and completed <= 0:
        entry["symbol_progress"] = "-"
    elif total <= 0:
        entry["symbol_progress"] = str(completed)
    else:
        entry["symbol_progress"] = f"{completed}/{total}"
    return entry


def _normalise_task_entry(entry: dict[str, T.Any]) -> dict[str, T.Any]:
    if not isinstance(entry, dict):
        return {}
    task_id = str(entry.get("id") or "")
    base = TASK_TEMPLATES.get(task_id, {"id": task_id})
    merged = {**base, **entry}
    if not merged.get("name"):
        merged["name"] = base.get("name", task_id)
    merged.setdefault("status", "等待")
    merged.setdefault("detail", "")
    merged.setdefault("updated_at", dt.datetime.now(US_EASTERN).strftime("%H:%M:%S"))
    merged.setdefault("start_time", base.get("start_time"))
    merged.setdefault("end_time", base.get("end_time"))
    return _ensure_task_progress(merged)


def _merge_task_updates(
    existing: dict[str, T.Any] | None,
    updates: list[dict[str, T.Any]],
    target_date: str | None = None,
) -> dict[str, T.Any]:
    current_map: dict[str, dict[str, T.Any]] = {}
    existing_target = None
    if isinstance(existing, dict):
        existing_target = str(existing.get("target_date") or "") or None
        tasks = existing.get("tasks")
        if isinstance(tasks, list):
            for item in tasks:
                if isinstance(item, dict) and item.get("id"):
                    current_map[str(item["id"])] = item

    requested_target = str(target_date) if target_date else None
    if requested_target and requested_target != existing_target:
        current_map = {}
        existing_target = requested_target
    elif requested_target is None:
        requested_target = existing_target

    now_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")

    for update in updates:
        norm = _normalise_task_entry(update)
        task_id = str(norm.get("id") or "")
        if not task_id:
            continue

        previous = current_map.get(task_id, {})

        status = norm.get("status") or previous.get("status") or "等待"
        norm["status"] = status

        total_symbols = norm.get("total_symbols")
        if total_symbols is None and isinstance(previous, dict):
            total_symbols = previous.get("total_symbols")
        completed_symbols = norm.get("completed_symbols")
        if completed_symbols is None and isinstance(previous, dict):
            completed_symbols = previous.get("completed_symbols")
        processed_symbols = norm.get("processed_symbols")
        if processed_symbols is None and isinstance(previous, dict):
            processed_symbols = previous.get("processed_symbols")

        if status == "进行中":
            start_val = norm.get("start_time") or previous.get("start_time") or now_ts
            norm["start_time"] = start_val
            if previous.get("end_time") and not norm.get("end_time"):
                norm["end_time"] = previous.get("end_time")
        elif status in {"已完成", "失败", "无数据"}:
            norm.setdefault("start_time", previous.get("start_time") or now_ts)
            norm.setdefault("end_time", now_ts)
        else:
            if previous.get("start_time") and not norm.get("start_time"):
                norm["start_time"] = previous.get("start_time")
            if previous.get("end_time") and not norm.get("end_time"):
                norm["end_time"] = previous.get("end_time")

        norm["total_symbols"] = total_symbols
        norm["completed_symbols"] = completed_symbols
        norm["processed_symbols"] = processed_symbols
        norm = _ensure_task_progress(norm)
        norm["updated_at"] = now_ts

        current_map[task_id] = norm

    ordered: list[dict[str, T.Any]] = []
    block_following = False
    for key in TASK_ORDER:
        if key in current_map:
            entry = current_map[key]
            status = str(entry.get("status") or "等待")
            if block_following and status not in {"已完成", "失败", "无数据"}:
                if status != "等待":
                    detail_text = str(entry.get("detail") or "")
                    if not detail_text.startswith("等待前序任务完成"):
                        prefix = "等待前序任务完成"
                        detail_text = f"{prefix}：{detail_text}" if detail_text else prefix
                    entry["detail"] = detail_text
                entry["status"] = "等待"
                entry.pop("start_time", None)
                if entry.get("end_time") and entry.get("status") != "已完成":
                    entry.pop("end_time", None)
                entry["updated_at"] = now_ts
            else:
                if status in {"进行中", "等待", "失败", "无数据"}:
                    block_following = True
            entry = _ensure_task_progress(entry)
            ordered.append(entry)
    for key, value in current_map.items():
        if key not in TASK_ORDER:
            ordered.append(_ensure_task_progress(value))

    return {"tasks": ordered, "target_date": requested_target}


def _load_earnings_cache_raw() -> dict[str, T.Any]:
    directory_payload = _load_archive_directory(EARNINGS_ARCHIVE_DIR)

    legacy_payload: dict[str, T.Any] = {}
    try:
        with EARNINGS_CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    if isinstance(data, dict):
        legacy_payload = {str(k): v for k, v in data.items()}

    combined: dict[str, T.Any] = {}
    combined.update(legacy_payload)
    combined.update(directory_payload)
    return combined


def _get_cached_earnings(date_value: dt.date) -> dict[str, T.Any] | None:
    key = date_value.isoformat()
    with CACHE_LOCK:
        cache = _load_earnings_cache_raw()
        entry = cache.get(key)
        if not isinstance(entry, dict):
            return None
        # deep copy via JSON round-trip to avoid accidental mutation
        try:
            return json.loads(json.dumps(entry))
        except (TypeError, ValueError):
            return None


def _store_cached_earnings(
    date_value: dt.date,
    row_data: list[dict[str, T.Any]],
    status: str,
    *,
    options_filter_applied: bool | None = None,
) -> None:
    key = date_value.isoformat()
    payload = {
        "rowData": row_data,
        "status": status,
        "generated_at": dt.datetime.now(US_EASTERN).isoformat(),
    }
    if options_filter_applied is not None:
        payload["options_filter_applied"] = bool(options_filter_applied)
    with CACHE_LOCK:
        archive_file = _earnings_archive_file(date_value)
        payload_to_write = payload
        if archive_file.exists():
            try:
                with archive_file.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict):
                    existing.update(payload_to_write)
                    payload_to_write = existing
            except (OSError, json.JSONDecodeError):
                payload_to_write = payload
        _write_archive_file(archive_file, payload_to_write)

        cache = _load_earnings_cache_raw()
        cache[key] = payload_to_write
        try:
            with EARNINGS_CACHE_PATH.open("w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass


def _load_prediction_archive_raw() -> dict[str, T.Any]:
    directory_payload = _load_archive_directory(PREDICTION_ARCHIVE_DIR)

    legacy_payload: dict[str, T.Any] = {}
    try:
        with PREDICTION_ARCHIVE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        payload = {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        legacy_payload = {str(k): v for k, v in payload.items()}

    combined: dict[str, T.Any] = {}
    combined.update(legacy_payload)
    combined.update(directory_payload)
    return combined


def _get_prediction_archive(date_value: dt.date) -> dict[str, T.Any] | None:
    key = date_value.isoformat()
    with PREDICTION_LOCK:
        archive_file = _prediction_archive_file(date_value)
        if archive_file.exists():
            try:
                with archive_file.open("r", encoding="utf-8") as fh:
                    file_entry = json.load(fh)
            except (OSError, json.JSONDecodeError):
                file_entry = None
            if isinstance(file_entry, dict):
                try:
                    return json.loads(json.dumps(file_entry))
                except (TypeError, ValueError):
                    return file_entry

        archive = _load_prediction_archive_raw()
        entry = archive.get(key)
        if not isinstance(entry, dict):
            return None
        try:
            return json.loads(json.dumps(entry))
        except (TypeError, ValueError):
            return None


def _store_prediction_results(
    target_date: dt.date,
    row_data: list[dict[str, T.Any]],
    results: list[dict[str, T.Any]],
    status: str,
) -> None:
    key = target_date.isoformat()
    payload = {
        "rowData": row_data,
        "results": results,
        "status": status,
        "generated_at": dt.datetime.now(US_EASTERN).isoformat(),
    }
    with PREDICTION_LOCK:
        archive_file = _prediction_archive_file(target_date)
        payload_to_write = payload
        if archive_file.exists():
            try:
                with archive_file.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict):
                    existing.update(payload_to_write)
                    payload_to_write = existing
            except (OSError, json.JSONDecodeError):
                payload_to_write = payload
        _write_archive_file(archive_file, payload_to_write)

        archive = _load_prediction_archive_raw()
        archive[key] = payload_to_write
        try:
            with PREDICTION_ARCHIVE_PATH.open("w", encoding="utf-8") as fh:
                json.dump(archive, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass


def _store_prediction_evaluation(target_date: dt.date, evaluation: dict[str, T.Any]) -> None:
    key = target_date.isoformat()
    with PREDICTION_LOCK:
        archive_file = _prediction_archive_file(target_date)
        entry: dict[str, T.Any]
        if archive_file.exists():
            try:
                with archive_file.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict):
                    entry = existing
                else:
                    entry = {}
            except (OSError, json.JSONDecodeError):
                entry = {}
        else:
            entry = {}
        entry["evaluation"] = evaluation
        _write_archive_file(archive_file, entry)

        archive = _load_prediction_archive_raw()
        archive[key] = entry
        try:
            with PREDICTION_ARCHIVE_PATH.open("w", encoding="utf-8") as fh:
                json.dump(archive, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass


def _load_symbol_meta_cache() -> dict[str, dict[str, T.Any]]:
    global SYMBOL_META_CACHE
    if SYMBOL_META_CACHE:
        return SYMBOL_META_CACHE
    try:
        with SYMBOL_META_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        SYMBOL_META_CACHE = {}
        return SYMBOL_META_CACHE
    except json.JSONDecodeError:
        SYMBOL_META_CACHE = {}
        return SYMBOL_META_CACHE

    if isinstance(payload, dict):
        SYMBOL_META_CACHE = {str(k).upper(): v for k, v in payload.items() if isinstance(v, dict)}
    else:
        SYMBOL_META_CACHE = {}
    return SYMBOL_META_CACHE


def _persist_symbol_meta_cache(cache: dict[str, dict[str, T.Any]]) -> None:
    SYMBOL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SYMBOL_META_PATH.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _fetch_symbol_profile(symbol: str) -> dict[str, T.Any] | None:
    url = "https://finnhub.io/api/v1/stock/profile2"
    try:
        resp = requests.get(
            url,
            params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY},
            timeout=10,
        )
        if not resp.ok:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    sector = str(data.get("finnhubIndustry") or data.get("sector") or "").strip()
    industry = str(data.get("industry") or data.get("gicsSector") or "").strip()
    name = str(data.get("name") or data.get("ticker") or "").strip()
    exchange = str(data.get("exchange") or data.get("exchangeSymbol") or "").strip()
    return {
        "sector": sector,
        "industry": industry,
        "name": name,
        "exchange": exchange,
        "source": "finnhub",
        "last_updated": dt.datetime.now(US_EASTERN).isoformat(),
    }


def _get_symbol_metadata(symbol: str) -> dict[str, T.Any]:
    symbol_key = symbol.upper()
    with SYMBOL_META_LOCK:
        cache = _load_symbol_meta_cache()
        entry = cache.get(symbol_key)
        if entry:
            return entry

    profile = _fetch_symbol_profile(symbol_key)
    if profile is None:
        return {}

    with SYMBOL_META_LOCK:
        cache = _load_symbol_meta_cache()
        cache[symbol_key] = profile
        _persist_symbol_meta_cache(cache)
    return profile


def _get_symbol_sector(symbol: str) -> str:
    metadata = _get_symbol_metadata(symbol)
    sector = metadata.get("sector") if isinstance(metadata, dict) else ""
    return sector or "未知"


def _coerce_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _date_picker_bounds() -> tuple[dt.date, dt.date, dt.date]:
    today = us_eastern_today()
    max_date = today + dt.timedelta(days=14)
    min_date = today - dt.timedelta(days=30)
    with CACHE_LOCK:
        cache = _load_earnings_cache_raw()
    cache_dates: list[dt.date] = []
    for key in cache.keys():
        if isinstance(key, str):
            try:
                cache_dates.append(dt.date.fromisoformat(key))
            except ValueError:
                continue
    if cache_dates:
        min_cache_date = min(cache_dates)
        if min_cache_date < min_date:
            min_date = min_cache_date
    return today, min_date, max_date


def _load_dci_payloads() -> dict[str, dict[str, T.Any]]:
    try:
        with DCI_DATA_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict[str, T.Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key.upper()] = value
    return out


def _extract_timeline_payload(
    data: dict[str, T.Any] | None,
    config: dict[str, T.Any],
) -> tuple[dict[str, T.Any] | None, str]:
    """返回指定时点配置对应的输入以及命中的别名。"""

    if not isinstance(data, dict):
        return None, ""

    search_buckets: list[dict[str, T.Any]] = []
    snapshots = data.get("snapshots")
    if isinstance(snapshots, dict):
        search_buckets.append(snapshots)

    search_buckets.append(data)

    aliases = [config.get("key")] + list(config.get("aliases", []))
    for bucket in search_buckets:
        for alias in aliases:
            if not alias:
                continue
            candidate = bucket.get(alias)
            if isinstance(candidate, dict):
                return candidate, str(alias)

    if config.get("key") == "decision_day" and isinstance(data.get("factors"), dict):
        return data, str(config.get("key"))

    return None, ""


def _compute_dci_for_symbols(
    symbols: list[str],
    metadata_map: dict[str, dict[str, T.Any]] | None = None,
    progress_callback: T.Optional[
        T.Callable[[str, dict[str, T.Any], str, T.Optional[str]], None]
    ] = None,
    timeline_configs: list[dict[str, T.Any]] | None = None,
) -> tuple[list[dict[str, T.Any]], list[str], list[str]]:
    payloads = _load_dci_payloads()
    results: list[dict[str, T.Any]] = []
    missing: list[str] = []
    errors: list[str] = []

    timelines = timeline_configs or PREDICTION_TIMELINES

    for raw_symbol in symbols:
        symbol = (raw_symbol or "").upper()
        if not symbol:
            continue
        data = payloads.get(symbol)
        if not data:
            missing.append(symbol)
            continue

        meta_entry = metadata_map.get(symbol) if isinstance(metadata_map, dict) else None
        decision_date = str((meta_entry or {}).get("decision_date") or "")
        bucket = str((meta_entry or {}).get("bucket") or "")
        company_name = str((meta_entry or {}).get("company") or "")
        sector_value = (meta_entry or {}).get("sector") or _get_symbol_sector(symbol)
        sector_key = sector_value if sector_value and sector_value != "未知" else None

        for timeline in timelines:
            if progress_callback:
                try:
                    progress_callback(symbol, timeline, "start", None)
                except Exception:  # pragma: no cover - logging best effort
                    pass

            payload, alias = _extract_timeline_payload(data, timeline)
            if not payload:
                if progress_callback:
                    try:
                        progress_callback(symbol, timeline, "missing", None)
                    except Exception:  # pragma: no cover - logging best effort
                        pass
                missing.append(f"{symbol}({timeline.get('label')})")
                continue

            try:
                inputs = build_inputs(symbol, payload)
                dci_result = compute_dci(inputs)

                rl_direction = ""
                rl_p_up_pct: float | None = None
                rl_delta_pct: float | None = None
                rl_prediction_id = ""
                selection_value = f"{symbol}::{timeline.get('key')}"

                if RL_MANAGER is not None:
                    try:
                        rl_pred = RL_MANAGER.record_prediction(
                            dci_result,
                            sector=sector_key,
                        )
                        rl_direction = "买入" if rl_pred.direction > 0 else "卖出"
                        rl_p_up_pct = round(rl_pred.adjusted_probability * 100.0, 2)
                        rl_delta_pct = round((rl_pred.adjusted_probability - dci_result.p_up) * 100.0, 2)
                        rl_prediction_id = rl_pred.prediction_id
                        selection_value = f"{symbol}::{rl_prediction_id}"
                    except Exception as rl_exc:  # pragma: no cover - best effort
                        errors.append(f"{symbol}({timeline.get('label')}): RL 调整失败 - {rl_exc}")

                shrink_map = {
                    key: round(float(value), 4)
                    for key, value in (dci_result.shrink_factors or {}).items()
                }
                scaled_map = {
                    key: round(float(value), 4)
                    for key, value in (dci_result.scaled_factors or {}).items()
                }
                factor_weight_map = {
                    key: round(float(value), 4)
                    for key, value in (dci_result.factor_weights or {}).items()
                }
                inputs_map = {
                    "z_cons": round(float(inputs.z_cons), 4),
                    "z_narr": round(float(inputs.z_narr), 4),
                    "CI": round(float(inputs.crowding_index), 2),
                    "Q": round(float(inputs.quality_score), 2),
                    "D": round(float(inputs.disagreement), 4),
                    "EM_pct": round(float(inputs.expected_move_pct), 2),
                    "S_stab": round(float(inputs.stability), 3),
                    "shock_flag": int(inputs.shock_flag),
                }

                results.append(
                    {
                        "symbol": symbol,
                        "timeline_label": timeline.get('label'),
                        "timeline_key": timeline.get('key'),
                        "timeline_alias": alias or timeline.get('key'),
                        "lookback_days": int(timeline.get('lookback', 0)),
                        "decision_date": decision_date,
                        "bucket": bucket,
                        "company": company_name,
                        "sector": sector_value or "未知",
                        "direction": "买入" if dci_result.direction > 0 else "卖出",
                        "p_up_pct": round(dci_result.p_up * 100.0, 2),
                        "dci_base": round(dci_result.dci_base, 2),
                        "dci_penalised": round(dci_result.dci_penalised, 2),
                        "dci_final": round(dci_result.dci_final, 2),
                        "position_weight": round(dci_result.position_weight, 2),
                        "position_bucket": dci_result.position_bucket,
                        "certainty": round(dci_result.certainty, 2),
                        "base_score": round(dci_result.base_score, 4),
                        "shrink_eg": shrink_map.get("shrink_EG"),
                        "shrink_ci": shrink_map.get("shrink_CI"),
                        "shrink_disagreement": shrink_map.get("disagreement"),
                        "shrink_shock": shrink_map.get("shock"),
                        "input_z_cons": inputs_map["z_cons"],
                        "input_z_narr": inputs_map["z_narr"],
                        "input_ci": inputs_map["CI"],
                        "input_q": inputs_map["Q"],
                        "input_d": inputs_map["D"],
                        "input_em_pct": inputs_map["EM_pct"],
                        "input_s_stab": inputs_map["S_stab"],
                        "input_shock_flag": inputs_map["shock_flag"],
                        "rl_direction": rl_direction,
                        "rl_p_up_pct": rl_p_up_pct,
                        "rl_delta_pct": rl_delta_pct,
                        "rl_prediction_id": rl_prediction_id,
                        "selection_value": selection_value,
                        "shrink_factors": shrink_map,
                        "scaled_factors": scaled_map,
                        "factor_weights": factor_weight_map,
                        "inputs": inputs_map,
                    }
                )
                if progress_callback:
                    try:
                        progress_callback(symbol, timeline, "success", None)
                    except Exception:  # pragma: no cover - logging best effort
                        pass
            except Exception as exc:  # pragma: no cover - defensive programming
                errors.append(f"{symbol}({timeline.get('label')}): {exc}")
                if progress_callback:
                    try:
                        progress_callback(symbol, timeline, "error", str(exc))
                    except Exception:  # pragma: no cover - logging best effort
                        pass

    results.sort(key=lambda row: (row.get("symbol", ""), row.get("lookback_days", 0)))
    return results, missing, errors


def _check_resource_connections(ft_session: dict[str, T.Any] | None) -> list[dict[str, T.Any]]:
    statuses: list[dict[str, T.Any]] = []

    def add_status(name: str, ok: bool, detail: str) -> None:
        statuses.append({"resource": name, "ok": ok, "detail": detail})

    today_str = dt.date.today().isoformat()

    def run_with_retry(checker: T.Callable[[], tuple[bool, str]], retries: int = 1) -> tuple[bool, str]:
        attempts = retries + 1
        details: list[str] = []
        for attempt in range(attempts):
            ok, detail = checker()
            if ok:
                if attempt:
                    suffix = f"（第{attempt + 1}次尝试成功）"
                    detail = f"{detail}{suffix}" if detail else suffix
                return True, detail
            details.append(detail or "")
        combined = "；".join(filter(None, details))
        if retries:
            retry_note = f"（已重试 {retries} 次）"
            combined = f"{combined}{retry_note}" if combined else retry_note
        return False, combined

    # Nasdaq earnings API
    def _check_nasdaq_once() -> tuple[bool, str]:
        try:
            resp = requests.get(
                NASDAQ_API,
                params={"date": today_str},
                headers=HEADERS,
                timeout=5,
            )
            ok = resp.ok
            detail = f"HTTP {resp.status_code}" if resp is not None else "无响应"
        except requests.RequestException as exc:
            ok = False
            detail = f"请求异常：{exc}"[:160]
        return ok, detail

    ok, detail = run_with_retry(_check_nasdaq_once)
    add_status("Nasdaq 财报 API", ok, detail)

    # Finnhub API
    def _check_finnhub_once() -> tuple[bool, str]:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": "AAPL", "token": FINNHUB_API_KEY},
                timeout=5,
            )
            if resp.ok:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                ok = isinstance(payload, dict) and "c" in payload
                detail = "返回有效报价" if ok else "返回内容异常"
            else:
                ok = False
                detail = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            ok = False
            detail = f"请求异常：{exc}"[:160]
        return ok, detail

    ok, detail = run_with_retry(_check_finnhub_once)
    add_status("Finnhub API", ok, detail)

    # OpenFIGI API
    def _check_openfigi_once() -> tuple[bool, str]:
        try:
            resp = requests.post(
                "https://api.openfigi.com/v3/mapping",
                headers={
                    "Content-Type": "application/json",
                    "X-OPENFIGI-APIKEY": OPENFIGI_API_KEY,
                },
                json=[{"idType": "TICKER", "idValue": "AAPL"}],
                timeout=5,
            )
            if resp.ok:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                ok = isinstance(payload, list) and bool(payload)
                detail = "返回有效映射" if ok else "返回内容异常"
            else:
                ok = False
                detail = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            ok = False
            detail = f"请求异常：{exc}"[:160]
        return ok, detail

    ok, detail = run_with_retry(_check_openfigi_once)
    add_status("OpenFIGI API", ok, detail)

    # FRED API
    def _check_fred_once() -> tuple[bool, str]:
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series",
                params={
                    "series_id": "DGS3MO",
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                },
                timeout=5,
            )
            if resp.ok:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                ok = isinstance(payload, dict) and payload.get("seriess")
                detail = "返回有效数据" if ok else "返回内容异常"
            else:
                ok = False
                detail = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            ok = False
            detail = f"请求异常：{exc}"[:160]
        return ok, detail

    ok, detail = run_with_retry(_check_fred_once)
    add_status("FRED API", ok, detail)

    # Firstrade session status
    if isinstance(ft_session, dict) and ft_session:
        sid = str(ft_session.get("sid", ""))
        ok = bool(ft_session.get("sid"))
        detail = f"已缓存会话 (sid {sid[:4]}...)" if ok else "会话信息不完整"
    else:
        ok = False
        detail = "尚未登录或无会话信息"
    add_status("Firstrade 会话", ok, detail)

    return statuses


def _render_connection_statuses(
    statuses: list[dict[str, T.Any]], checked_at: str | None = None
) -> T.Union[str, dbc.Table, html.Div]:
    timestamp_block = None
    if checked_at:
        timestamp_block = html.Div(
            f"最后检查时间：{checked_at}",
            style={"marginBottom": "6px", "fontWeight": "bold"},
        )

    if not statuses:
        message = html.Div("暂无连接状态信息。")
        if timestamp_block:
            return html.Div([timestamp_block, message])
        return message

    rows: list[html.Tr] = []
    for entry in statuses:
        ok = bool(entry.get("ok"))
        badge = dbc.Badge(
            "正常" if ok else "异常",
            color="success" if ok else "danger",
            pill=True,
        )
        rows.append(
            html.Tr(
                [
                    html.Td(entry.get("resource", "")),
                    html.Td(badge),
                    html.Td(entry.get("detail", "")),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("资源"), html.Th("状态"), html.Th("详情")])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        className="mt-2",
    )
    content: list[T.Any] = []
    if timestamp_block:
        content.append(timestamp_block)
    content.append(table)
    return html.Div(content)


def _init_run_state(run_id: str, target_date: dt.date) -> None:
    with RUN_LOCK:
        RUN_STATES[run_id] = {
            "logs": [],
            "rowData": None,
            "status": "",
            "session_state": NO_UPDATE_SENTINEL,
            "completed": False,
            "target_date": target_date,
        }


def _get_run_state(run_id: str) -> dict[str, T.Any] | None:
    with RUN_LOCK:
        state = RUN_STATES.get(run_id)
        if not state:
            return None
        out = dict(state)
        out["logs"] = list(state.get("logs", []))
        return out


def _append_run_log(run_id: str, message: str) -> None:
    stamp = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
    entry = f"[{stamp}] {message}"
    with RUN_LOCK:
        state = RUN_STATES.get(run_id)
        if not state:
            return
        state.setdefault("logs", []).append(entry)


def _update_run_state(run_id: str, **kwargs: T.Any) -> None:
    with RUN_LOCK:
        state = RUN_STATES.get(run_id)
        if not state:
            return
        state.update(kwargs)

# ---------- Nasdaq API ----------
NASDAQ_API = "https://api.nasdaq.com/api/calendar/earnings"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/earnings",
}


def us_eastern_today() -> dt.date:
    return dt.datetime.now(US_EASTERN).date()


def this_friday(base_date: dt.date) -> dt.date:
    wd = base_date.isoweekday()  # Mon=1 ... Sun=7
    if wd <= 5:
        return base_date + dt.timedelta(days=5 - wd)
    return base_date + dt.timedelta(days=(12 - wd))


def next_trading_day(base_date: dt.date) -> dt.date:
    """Return the next US trading day (skipping weekends)."""

    wd = base_date.weekday()  # Mon=0 ... Sun=6
    if wd >= 4:  # Friday, Saturday, Sunday -> jump to Monday
        return base_date + dt.timedelta(days=(7 - wd))
    return base_date + dt.timedelta(days=1)


def previous_trading_day(base_date: dt.date) -> dt.date:
    """Return the previous US trading day (skipping weekends)."""

    wd = base_date.weekday()  # Mon=0 ... Sun=6
    if wd == 0:  # Monday -> previous Friday
        return base_date - dt.timedelta(days=3)
    if wd == 6:  # Sunday -> previous Friday
        return base_date - dt.timedelta(days=2)
    if wd == 5:  # Saturday -> previous Friday
        return base_date - dt.timedelta(days=1)
    return base_date - dt.timedelta(days=1)


def fetch_earnings_by_date(d: dt.date) -> T.List[dict]:
    for _ in range(2):  # simple retry
        r = requests.get(NASDAQ_API, params={"date": d.strftime("%Y-%m-%d")},
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        try:
            j = r.json()
        except json.JSONDecodeError:
            time.sleep(1)
            continue

        data = j.get("data") if isinstance(j, dict) else None
        rows = None
        if isinstance(data, dict):
            cal = data.get("calendar") if isinstance(data.get("calendar"), dict) else None
            rows = (cal or data).get("rows") if isinstance((cal or data), dict) else None
        if rows is None and isinstance(j, dict):
            rows = j.get("rows")
        if not rows:
            return []

        out = []
        for row in rows:
            symbol = (row.get("symbol") or row.get("Symbol") or row.get("companyTickerSymbol") or "").strip()
            time_field = (row.get("time") or row.get("Time") or row.get("EPSTime") or row.get("timeStatus") or row.get("when") or "").strip()
            name = (row.get("companyName") or row.get("name") or row.get("Company") or "").strip()
            if symbol:
                out.append({
                    "symbol": symbol.upper(),
                    "company": name,
                    "time": time_field,
                    "raw": row,
                })
        return out
    return []


# ---------- Time bucket filters ----------

def _is_after_hours(s: str) -> bool:
    s = (s or "").lower()
    return ("after" in s) or ("post" in s) or ("amc" in s) or ("after-hours" in s)


def _is_pre_market(s: str) -> bool:
    text = s or ""
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("pre", "before", "bmo", "pre-market")):
        return True
    if any(keyword in lowered for keyword in ("tbd", "unconfirmed")):
        return True
    if any(keyword in text for keyword in ("待定", "未确认", "待公布")):
        return True
    return False


def _is_time_not_supplied(s: str) -> bool:
    text = s or ""
    if _is_pre_market(text):
        return False
    lowered = text.lower()
    return ("not" in lowered and "suppl" in lowered) or ("tbd" in lowered) or ("unconfirmed" in lowered) or (lowered == "")


def pick_by_times(rows: T.List[dict], want_after_hours=False, want_pre_market=False, want_not_supplied=False) -> T.List[dict]:
    out = []
    for r in rows:
        t = r.get("time", "")
        ok = (
            (want_after_hours and _is_after_hours(t)) or
            (want_pre_market and _is_pre_market(t)) or
            (want_not_supplied and _is_time_not_supplied(t))
        )
        if ok:
            out.append(r)
    return out


def append_log(logs: T.Optional[T.List[str]], message: str) -> T.List[str]:
    if not isinstance(logs, list):
        logs = []
    stamp = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
    return logs + [f"[{stamp}] {message}"]
# ---------- Firstrade (unofficial) ----------
class FTClient:
    LOGIN_URL = "https://api3x.firstrade.com/sess/login"
    VERIFY_URL = "https://api3x.firstrade.com/sess/verify_pin"
    OPTIONS_URL = "https://api3x.firstrade.com/public/oc"
    OPTION_QUOTE_URL = "https://api3x.firstrade.com/market/quote/option"
    OPTION_PREVIEW_URL = "https://api3x.firstrade.com/trade/options/preview"
    OPTION_PLACE_URL = "https://api3x.firstrade.com/trade/options/place"
    SESSION_HEADERS = {
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
        "User-Agent": "okhttp/4.9.2",
        "access-token": "833w3XuIFycv18ybi",
    }

    def __init__(
        self,
        username: str,
        password: str,
        twofa_code: str | None = None,
        session_state: dict[str, T.Any] | None = None,
        login: bool = True,
        logger: T.Optional[T.Callable[[str], None]] = None,
    ):
        self.username = username or ""
        self.password = password or ""
        self.twofa_code = (twofa_code or "").strip()
        self.session: T.Optional[requests.Session] = None
        self.enabled = False
        self.error: str | None = None
        self._login_json: dict[str, T.Any] | None = None
        self.session_state: dict[str, T.Any] = {}
        self._logger = logger
        self.account_ids: list[str] = []
        self._default_host = urlparse(self.LOGIN_URL).netloc

        if session_state:
            self._restore_session(session_state)
        elif login:
            self._init()

    def _log(self, message: str) -> None:
        if self._logger:
            try:
                self._logger(message)
            except Exception:
                pass

    def _init(self) -> None:
        try:
            sess = requests.Session()
            sess.headers.update(self.SESSION_HEADERS)
            if self._default_host:
                sess.headers["Host"] = self._default_host
            self._log("正在初始化 Firstrade 会话并发送登录请求。")
            login_resp = sess.post(
                self.LOGIN_URL,
                data={"username": self.username, "password": self.password},
                timeout=20,
            )
            login_data = self._safe_json(login_resp)
            self._login_json = login_data

            if login_resp.status_code != 200:
                detail = ""
                if isinstance(login_data, dict):
                    detail = str(login_data.get("message") or login_data.get("error") or "").strip()
                self._log(f"Firstrade 登录出现 HTTP {login_resp.status_code}{f'（{detail}）' if detail else ''} 错误。")
                self.error = f"登录 HTTP {login_resp.status_code}{f'（{detail}）' if detail else ''}"
                return
            if not isinstance(login_data, dict):
                self._log("Firstrade 登录返回异常数据。")
                self.error = "登录响应异常"
                return

            err_msg = str(login_data.get("error") or "").strip()
            if err_msg:
                self._log(f"Firstrade 登录返回错误：{err_msg}")
                self.error = err_msg
                return

            sid = login_data.get("sid")
            ftat = login_data.get("ftat")
            t_token = login_data.get("t_token")
            verification_sid = login_data.get("verificationSid")
            requires_mfa = bool(login_data.get("mfa"))

            if sid:
                sess.headers["sid"] = sid
                self._log("已从 Firstrade 登录响应中获取会话 ID。")

            if requires_mfa:
                self._log("Firstrade 登录需要多重验证。")
                if not self.twofa_code:
                    self.error = "Firstrade 登录需要输入双重验证码（短信或动态口令）。"
                    return

                verify_payload: dict[str, T.Any] = {
                    "remember_for": "30",
                }
                if t_token:
                    verify_payload["t_token"] = t_token

                # SMS/email style MFA exposes verificationSid; authenticator-only accounts do not.
                code = self.twofa_code
                if verification_sid:
                    sess.headers["sid"] = verification_sid
                    self._log("使用验证 SID 进行一次性验证码校验。")
                    verify_payload["verificationSid"] = verification_sid
                    verify_payload["otpCode"] = code
                else:
                    self._log("提交认证器验证码进行校验。")
                    verify_payload["mfaCode"] = code

                verify_resp = sess.post(self.VERIFY_URL, data=verify_payload, timeout=20)
                verify_data = self._safe_json(verify_resp)
                if verify_resp.status_code != 200:
                    detail = ""
                    if isinstance(verify_data, dict):
                        detail = str(verify_data.get("message") or verify_data.get("error") or "").strip()
                    self._log(
                        f"Firstrade 多重验证失败，HTTP {verify_resp.status_code}{f'（{detail}）' if detail else ''}。"
                    )
                    self.error = f"双重验证 HTTP {verify_resp.status_code}{f'（{detail}）' if detail else ''}"
                    return
                if not isinstance(verify_data, dict):
                    self._log("Firstrade 多重验证返回异常数据。")
                    self.error = "双重验证响应异常"
                    return
                err_msg = str(verify_data.get("error") or "").strip()
                if err_msg:
                    self._log(f"Firstrade 多重验证报错：{err_msg}")
                    self.error = err_msg
                    return
                ftat = verify_data.get("ftat", ftat)
                sid = verify_data.get("sid") or verify_data.get("verificationSid") or verification_sid or sid

            if ftat:
                sess.headers["ftat"] = ftat
            if sid:
                sess.headers["sid"] = sid

            self.session = sess
            self.enabled = bool(ftat and sid)
            if self.enabled:
                self._update_accounts(login_data)
                self.session_state = {
                    "ftat": ftat,
                    "sid": sid,
                    "timestamp": time.time(),
                    "accounts": list(self.account_ids),
                }
                self._log("Firstrade 会话建立成功。")
            if not self.enabled and not self.error:
                self._log("Firstrade 会话缺少 SID/FTAT。")
                self.error = "Firstrade 会话缺少 SID/FTAT"
        except Exception as exc:
            self._log(f"Firstrade 登录出现意外错误：{exc}")
            self.error = str(exc)
            return

    def _restore_session(self, session_state: dict[str, T.Any]) -> None:
        ftat = session_state.get("ftat")
        sid = session_state.get("sid")
        if not ftat or not sid:
            self._log("无法恢复 Firstrade 会话：缺少 SID/FTAT。")
            self.error = "缺少会话令牌"
            self.enabled = False
            self.session_state = {}
            return

        sess = requests.Session()
        sess.headers.update(self.SESSION_HEADERS)
        host = urlparse(self.LOGIN_URL).netloc or self._default_host
        if host:
            sess.headers["Host"] = host
        sess.headers["ftat"] = ftat
        sess.headers["sid"] = sid
        self.session = sess
        self.enabled = True
        self.error = None
        self.session_state = {
            "ftat": ftat,
            "sid": sid,
            "timestamp": session_state.get("timestamp", time.time()),
            "accounts": list(session_state.get("accounts") or []),
        }
        self.account_ids = [
            str(a).strip()
            for a in session_state.get("accounts", [])
            if isinstance(a, str) and str(a).strip()
        ]
        self._log("已通过缓存令牌恢复 Firstrade 会话。")

    @staticmethod
    def _safe_json(resp: requests.Response) -> T.Any:
        try:
            return resp.json()
        except Exception:
            return None

    def export_session_state(self) -> dict[str, T.Any]:
        state = dict(self.session_state)
        if self.account_ids:
            state.setdefault("accounts", list(self.account_ids))
        return state

    def _update_accounts(self, payload: dict[str, T.Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        accounts: list[str] = []

        def add(value: T.Any) -> None:
            if value is None:
                return
            text = str(value).strip()
            if text and text not in accounts:
                accounts.append(text)

        candidate_keys = [
            "accounts",
            "accountList",
            "acctList",
            "account_list",
            "acctNoList",
            "data",
        ]
        for key in candidate_keys:
            items = payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        add(item.get("account"))
                        add(item.get("acctNo"))
                        add(item.get("accountNumber"))
                    else:
                        add(item)

        for key in ("account", "acctNo", "accountNumber", "primaryAccount"):
            add(payload.get(key))

        if accounts:
            self.account_ids = accounts

    def has_weekly_expiring_on(self, symbol: str, expiry: dt.date) -> T.Optional[bool]:
        if not self.enabled or not self.session:
            return None

        try:
            self._log(f"正在请求 {symbol.upper()} 的周度期权到期日。")
            resp = self.session.get(
                self.OPTIONS_URL,
                params={"m": "get_exp_dates", "root_symbol": symbol.upper()},
                timeout=20,
            )
            data = self._safe_json(resp)
            if resp.status_code == 401:
                self.enabled = False
                self.error = "Firstrade 会话已过期"
                self._log("查询期权时 Firstrade 会话已过期（HTTP 401）。")
                return None
            if resp.status_code != 200 or not isinstance(data, dict):
                self._log(f"查询 {symbol.upper()} 的期权失败，HTTP {resp.status_code}。")
                return None
            err_msg = str(data.get("error") or "").strip()
            if err_msg:
                self._log(f"查询 {symbol.upper()} 的期权返回错误：{err_msg}")
                return None
            items = data.get("items")
            if not isinstance(items, list):
                return None

            target = expiry.strftime("%Y%m%d")
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                exp_date = str(entry.get("exp_date", "")).strip()
                if exp_date != target:
                    continue
                exp_type = str(entry.get("exp_type", "")).upper()
                # exp_type "W" => weekly, "M" => monthly, other codes possible
                self._log(
                    f"找到 {symbol.upper()} 在 {target} 的到期日 {exp_date}（类型 {exp_type}）。"
                )
                return exp_type == "W"
            self._log(f"未找到 {symbol.upper()} 在 {target} 的匹配到期日。")
            return False
        except Exception:
            self._log(f"获取 {symbol.upper()} 期权到期日时出现异常。")
            return None



# ---------- Dash app ----------
AG_GRID_STYLES = [
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css",
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine.css",
]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP] + AG_GRID_STYLES)
app.title = "财报博弈自动化平台"

# Pre-fill username if you want; password intentionally left blank
DEFAULT_USERNAME = os.getenv("FTD_USERNAME", "sunweicheng")
DEFAULT_PASSWORD = os.getenv("FTD_PASSWORD", "Swc_661199")
DEFAULT_PICKER_DATE, MIN_PICKER_DATE, MAX_PICKER_DATE = _date_picker_bounds()
DEFAULT_PICKER_DATE_STR = DEFAULT_PICKER_DATE.isoformat()

initial_snapshot = RL_MANAGER.snapshot() if RL_MANAGER else None
layout_config = LayoutConfig(
    default_username=DEFAULT_USERNAME,
    default_password=DEFAULT_PASSWORD,
    default_picker_date=DEFAULT_PICKER_DATE,
    min_picker_date=MIN_PICKER_DATE,
    max_picker_date=MAX_PICKER_DATE,
    default_picker_date_str=DEFAULT_PICKER_DATE_STR,
    rl_snapshot=initial_snapshot,
    app_title=app.title,
    navbar_title="财报博弈自动化平台",
    main_heading="",
    prediction_timelines=PREDICTION_TIMELINES,
)

app.layout = build_layout(layout_config)



def build_targets(target_date: dt.date) -> pd.DataFrame:
    tomorrow_us = next_trading_day(target_date)
    is_friday = target_date.weekday() == 4
    pre_market_label = "下周一盘前" if is_friday else "次日盘前"
    not_supplied_label = "下周一待定" if is_friday else "次日待定"

    rows_today = fetch_earnings_by_date(target_date)
    rows_tomorrow = fetch_earnings_by_date(tomorrow_us)

    targets: list[dict] = []
    targets += [
        {**r, "decision_date": target_date.isoformat(), "bucket": "今日盘后"}
        for r in pick_by_times(rows_today, want_after_hours=True)
    ]
    targets += [
        {
            **r,
            "decision_date": target_date.isoformat(),
            "bucket": pre_market_label,
        }
        for r in pick_by_times(rows_tomorrow, want_pre_market=True)
    ]
    targets += [
        {
            **r,
            "decision_date": tomorrow_us.isoformat(),
            "bucket": not_supplied_label,
        }
        for r in pick_by_times(rows_tomorrow, want_not_supplied=True)
    ]

    # dedupe by (symbol, bucket, decision_date)
    seen = set()
    uniq = []
    for r in targets:
        key = (r["symbol"], r["bucket"], r["decision_date"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    df = (
        pd.DataFrame(uniq, columns=["symbol", "company", "bucket", "decision_date"])
        .sort_values(["bucket", "symbol"])
        .reset_index(drop=True)
    )
    return df


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
    logs = append_log(log_state, f"收到用户“{username or '（空）'}”的登录请求。")

    def log(message: str) -> None:
        nonlocal logs
        logs = append_log(logs, message)

    if not username or not password:
        log("登录终止：必须填写用户名和密码。")
        return no_update, "Firstrade 登录失败：请填写用户名和密码。", logs

    log("正在尝试登录 Firstrade……")
    ft = FTClient(username=username, password=password, twofa_code=twofa if twofa else None, logger=log)
    if ft.enabled:
        state = ft.export_session_state()
        msg = f"Firstrade 登录成功：会话 {state.get('sid', '')[:4]}..."
        log("Firstrade 登录成功。")
        return state, msg, logs

    log(f"Firstrade 登录失败：{ft.error or '未知错误'}。")
    return {}, f"Firstrade 登录失败：{ft.error or '未知错误'}", logs


@app.callback(
    Output("run-id-store", "data"),
    Output("log-store", "data", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("table", "rowData", allow_duplicate=True),
    Output("log-poller", "disabled", allow_duplicate=True),
    Output("task-store", "data", allow_duplicate=True),
    Input("auto-run-trigger", "n_intervals"),
    Input("earnings-date-picker", "date"),
    Input("earnings-refresh-btn", "n_clicks"),
    Input("ft-session-store", "data"),
    State("ft-username", "value"),
    State("ft-password", "value"),
    State("ft-2fa", "value"),
    State("task-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def start_run(auto_intervals, selected_date, refresh_clicks, session_data, username, password, twofa, task_state):  # noqa: D401
    trigger = ctx.triggered_id
    session_state = session_data if isinstance(session_data, dict) else {}
    manual_refresh = trigger == "earnings-refresh-btn"
    login_trigger = trigger == "ft-session-store"
    if trigger == "auto-run-trigger":
        if auto_intervals is None:
            return no_update, no_update, no_update, no_update, no_update, no_update
    elif trigger == "earnings-date-picker":
        if not selected_date:
            return no_update, no_update, no_update, no_update, no_update, no_update
    elif manual_refresh:
        if not refresh_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update
    elif login_trigger:
        if not session_state:
            return no_update, no_update, no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update

    target_date = _coerce_date(selected_date) or us_eastern_today()
    today_limit = us_eastern_today()
    max_allowed = today_limit + dt.timedelta(days=14)
    if target_date > max_allowed:
        target_date = max_allowed
    min_allowed = MIN_PICKER_DATE
    if target_date < min_allowed:
        target_date = min_allowed

    target_date_iso = target_date.isoformat()

    cache_entry = _get_cached_earnings(target_date)
    bypass_cache = manual_refresh
    if login_trigger:
        if not cache_entry:
            return no_update, no_update, no_update, no_update, no_update, no_update
        options_flag = cache_entry.get("options_filter_applied")
        cached_status = str(cache_entry.get("status") or "")
        already_filtered = options_flag is True or (
            options_flag is None and "未进行周五期权筛选" not in cached_status
        )
        if already_filtered:
            return no_update, no_update, no_update, no_update, no_update, no_update
        bypass_cache = True

    if cache_entry and not bypass_cache:
        logs = append_log([], f"命中 {target_date} 的本地缓存。")
        cached_status = str(cache_entry.get("status") or "")
        extra = "（来自本地缓存，无需重新请求。）"
        status_out = (cached_status + "\n" + extra) if cached_status else extra
        row_data = cache_entry.get("rowData")
        if not isinstance(row_data, list):
            row_data = []
        archive_entry = _get_prediction_archive(target_date)
        timeline_has_results: set[str] = set()
        if isinstance(archive_entry, dict):
            raw_results = archive_entry.get("results")
            if isinstance(raw_results, list):
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    timeline_key = str(item.get("timeline_key") or "")
                    if timeline_key:
                        timeline_has_results.add(timeline_key)
        timeline_updates = []
        for cfg in PREDICTION_TIMELINES:
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            timeline_key = str(cfg.get("key") or "")
            has_result = timeline_key in timeline_has_results
            timeline_updates.append(
                {
                    "id": task_id,
                    "name": name,
                    "status": "已完成" if has_result else "等待",
                    "detail": (
                        f"缓存数据 -> {offset_label} 预测已完成"
                        if has_result
                        else f"缓存数据 -> {offset_label} 预测待运行"
                    ),
                }
            )

        tasks = _merge_task_updates(
            task_state,
            timeline_updates,
            target_date=target_date_iso,
        )
        return None, logs, status_out, row_data, True, tasks

    run_id = uuid.uuid4().hex
    _init_run_state(run_id, target_date)

    thread = threading.Thread(
        target=_execute_run,
        args=(
            run_id,
            username or "",
            password or "",
            (twofa or ""),
            session_state,
            target_date,
        ),
        daemon=True,
    )
    thread.start()

    if login_trigger and bypass_cache:
        status_message = f"检测到 Firstrade 登录，开始重新筛选 {target_date} 的数据……"
    elif manual_refresh:
        status_message = f"正在刷新 {target_date} 的财报列表……"
    else:
        status_message = f"开始获取 {target_date} 的数据……"
    waiting_detail = "等待财报列表完成后执行"
    if login_trigger and bypass_cache:
        waiting_detail = f"检测到登录，重新筛选 {target_date} 的数据后执行"
    elif manual_refresh:
        waiting_detail = f"手动刷新 -> 正在获取 {target_date} 的数据"
    else:
        waiting_detail = f"正在获取 {target_date} 的数据"

    timeline_waiting = []
    for idx, cfg in enumerate(PREDICTION_TIMELINES, start=1):
        task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
        detail_text = waiting_detail if idx == 1 else f"等待前序任务完成后执行 {offset_label}"
        timeline_waiting.append(
            {
                "id": task_id,
                "name": name,
                "status": "等待",
                "detail": detail_text,
            }
        )

    tasks = _merge_task_updates(
        task_state,
        timeline_waiting,
        target_date=target_date_iso,
    )
    return run_id, [], status_message, [], False, tasks


def _prepare_earnings_dataset(
    target_date: dt.date,
    session_state: dict[str, T.Any] | None,
    username: str,
    password: str,
    twofa: str,
    logger: T.Callable[[str], None] | None = None,
) -> tuple[list[dict[str, T.Any]], str | None, dict[str, T.Any] | None, bool, str | None]:
    """获取并筛选指定日期的财报列表。"""

    def _log(message: str) -> None:
        if logger:
            try:
                logger(message)
            except Exception:  # pragma: no cover - logging best effort
                pass

    actual_today = us_eastern_today()
    tomorrow_target = next_trading_day(target_date)
    fri = this_friday(target_date)
    _log(f"已开始获取 {target_date} 的数据。")

    try:
        df = build_targets(target_date)
        _log(f"从 Nasdaq API 获取到 {len(df)} 个标的。")
        for idx, row in df.iterrows():
            _log(f"准备标的 #{idx + 1}：{row['symbol']}（{row['bucket']}）")
    except Exception as exc:  # pragma: no cover - network errors
        _log(f"获取 Nasdaq 数据时出错：{exc}")
        return [], None, session_state or {}, False, f"获取 Nasdaq 数据失败：{exc}"

    attempted_ft = False
    ft_ok = False
    ft_error: str | None = None
    store_out: dict[str, T.Any] | None = None
    session_state = session_state or {}
    options_filter_applied = False

    if session_state and username and password:
        attempted_ft = True
        _log("使用已保存的 Firstrade 会话检查周度期权。")
        try:
            ft = FTClient(
                username=username,
                password=password,
                twofa_code=(twofa or None),
                session_state=session_state,
                login=False,
                logger=_log,
            )
            ft_ok = ft.enabled
            ft_error = ft.error
            if ft_ok:
                _log(f"开始为 {len(df)} 个标的查询周度期权。")
                weekly: list[T.Optional[bool]] = []
                symbols = df["symbol"].tolist()
                for idx, sym in enumerate(symbols, start=1):
                    _log(f"检查标的 #{idx}：{sym} 是否有周五到期期权")
                    try:
                        has_w = ft.has_weekly_expiring_on(sym, fri)
                    except Exception:  # pragma: no cover - network errors
                        has_w = None
                    weekly.append(has_w)
                df["weekly_exp_this_fri"] = weekly
                if weekly:
                    options_filter_applied = any(value is not None for value in weekly)
                else:
                    options_filter_applied = True
                _log("周度期权查询完成。")
                exported = ft.export_session_state() or session_state
                store_out = exported if exported else session_state
            else:
                df["weekly_exp_this_fri"] = None
                _log(f"Firstrade 会话失效：{ft_error or '未知错误'}，已清除缓存。")
                store_out = {}
        except Exception as exc:  # pragma: no cover - network errors
            ft_error = str(exc)
            df["weekly_exp_this_fri"] = None
            _log(f"查询 Firstrade 数据时出错：{ft_error}。")
            store_out = {}
    else:
        df["weekly_exp_this_fri"] = None
        if session_state:
            _log("检测到 Firstrade 会话但缺少凭证，跳过周度期权检查。")
        else:
            _log("未保存 Firstrade 会话，跳过周度期权检查。")

    if "weekly_exp_this_fri" in df.columns:
        before_count = len(df)
        df = df[df["weekly_exp_this_fri"].ne(False)].reset_index(drop=True)
        removed = before_count - len(df)
        if removed:
            _log(f"过滤掉 {removed} 个无周五到期期权的标的。")
        df = df.drop(columns=["weekly_exp_this_fri"])

    status_lines = [
        f"目标财报日：{target_date} | 下一交易日：{tomorrow_target}",
        f"美东当前日期：{actual_today} | 本周五：{fri}",
        f"符合条件的标的数量：{len(df)}",
    ]
    if attempted_ft:
        detail = ""
        if not ft_ok and ft_error:
            detail = f"（{ft_error}）"
        status_lines.append(f"Firstrade 会话状态：{'有效' if ft_ok else '失败'}{detail}")
    else:
        status_lines.append("Firstrade 会话状态：未登录（请先在连接页登录）")

    if not options_filter_applied:
        status_lines.append("提示：当前公司列表未进行周五期权筛选，请登录 Firstrade 并刷新。")

    row_records = df.to_dict("records")
    status_text = "\n".join(status_lines)

    if store_out is None:
        store_out = session_state

    return row_records, status_text, store_out, options_filter_applied, None


def _execute_run(
    run_id: str,
    username: str,
    password: str,
    twofa: str,
    session_state: dict[str, T.Any],
    target_date: dt.date,
) -> None:
    _update_run_state(run_id, status="正在请求 Nasdaq 财报数据……")

    rows, status_text, session_out, options_filter_applied, error_msg = _prepare_earnings_dataset(
        target_date,
        session_state,
        username,
        password,
        twofa,
        logger=lambda message: _append_run_log(run_id, message),
    )

    if error_msg:
        status_msg = f"【错误】{error_msg}"
        _update_run_state(
            run_id,
            rowData=[],
            status=status_msg,
            session_state=NO_UPDATE_SENTINEL,
            completed=True,
        )
        return

    _update_run_state(
        run_id,
        rowData=rows,
        status=status_text or "",
        session_state=session_out if session_out is not None else NO_UPDATE_SENTINEL,
        completed=True,
        options_filter_applied=options_filter_applied,
    )
    _store_cached_earnings(
        target_date,
        rows,
        status_text or "",
        options_filter_applied=options_filter_applied,
    )


@app.callback(
    Output("connection-status-area", "children"),
    Input("check-connections-btn", "n_clicks"),
    Input("connection-poller", "n_intervals"),
    State("ft-session-store", "data"),
)
def refresh_connection_status(n_clicks, poll_intervals, ft_session):  # noqa: D401
    del n_clicks
    del poll_intervals
    session_data = ft_session if isinstance(ft_session, dict) else None
    statuses = _check_resource_connections(session_data)
    checked_at = dt.datetime.now(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S")
    return _render_connection_statuses(statuses, checked_at)


@app.callback(
    Output("log-store", "data", allow_duplicate=True),
    Output("table", "rowData", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("ft-session-store", "data", allow_duplicate=True),
    Output("log-poller", "disabled", allow_duplicate=True),
    Output("task-store", "data", allow_duplicate=True),
    Input("log-poller", "n_intervals"),
    State("run-id-store", "data"),
    State("log-store", "data"),
    State("table", "rowData"),
    State("task-store", "data"),
    prevent_initial_call=True,
)
def poll_run_state(n_intervals, run_id, existing_logs, current_rows, task_state):  # noqa: D401
    del n_intervals
    if not run_id:
        return no_update, no_update, no_update, no_update, True, no_update

    existing_logs = existing_logs or []
    current_rows = current_rows or []

    state = _get_run_state(run_id)
    if not state:
        return no_update, no_update, no_update, no_update, True, no_update

    logs = state.get("logs", [])
    row_data = state.get("rowData")
    status = state.get("status")
    completed = state.get("completed", False)
    session_state = state.get("session_state", NO_UPDATE_SENTINEL)

    logs_out = logs if logs != existing_logs else no_update
    table_out = row_data if row_data is not None and row_data != current_rows else no_update
    status_out = status if status else no_update
    if session_state is NO_UPDATE_SENTINEL:
        session_out = no_update
    else:
        session_out = session_state

    task_updates: list[dict[str, T.Any]] = []
    target_date = state.get("target_date")
    target_date_str = target_date.isoformat() if isinstance(target_date, dt.date) else None
    if isinstance(status, str) and "错误" in status and isinstance(target_date, dt.date):
        for cfg in PREDICTION_TIMELINES:
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            task_updates.append(
                {
                    "id": task_id,
                    "name": name,
                    "status": "失败",
                    "detail": f"财报列表失败，无法执行 {offset_label}",
                }
            )
    if isinstance(row_data, list) and row_data != current_rows and isinstance(target_date, dt.date):
        options_flag = state.get("options_filter_applied")
        base_detail = f"财报列表就绪，共 {len(row_data)} 个标的"
        if options_flag is False:
            base_detail += "（未进行周五期权筛选）"
        for idx, cfg in enumerate(PREDICTION_TIMELINES, start=1):
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            if idx == 1:
                detail_text = base_detail
            else:
                detail_text = f"等待前序任务完成后执行 {offset_label}"
            task_updates.append(
                {
                    "id": task_id,
                    "name": name,
                    "status": "等待",
                    "detail": detail_text,
                }
            )

    tasks_out = (
        _merge_task_updates(task_state, task_updates, target_date=target_date_str)
        if task_updates
        else no_update
    )

    return logs_out, table_out, status_out, session_out, bool(completed), tasks_out


@app.callback(
    Output("prediction-store", "data"),
    Output("prediction-status", "children"),
    Output("rl-agent-store", "data"),
    Output("task-store", "data", allow_duplicate=True),
    Output("log-store", "data", allow_duplicate=True),
    Input("table", "selectedRows"),
    Input("table", "rowData"),
    State("task-store", "data"),
    State("earnings-date-picker", "date"),
    State("log-store", "data"),
    State("ft-session-store", "data"),
    State("ft-username", "value"),
    State("ft-password", "value"),
    State("ft-twofa", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_predictions(
    selected_rows,
    row_data,
    task_state,
    picker_date,
    log_state,
    session_state,
    username,
    password,
    twofa,
):  # noqa: D401
    initial_snapshot = RL_MANAGER.snapshot() if RL_MANAGER is not None else None
    triggered = ctx.triggered_id
    task_state_local = task_state if isinstance(task_state, dict) else {"tasks": []}
    tasks_changed = False

    if triggered == "table.selectedRows" and not selected_rows:
        return no_update, no_update, no_update, no_update, no_update

    use_all_rows = triggered == "table.rowData"

    session_data = session_state if isinstance(session_state, dict) else {}
    username = (username or "").strip()
    password = (password or "").strip()
    twofa = (twofa or "").strip()

    table_rows = row_data if isinstance(row_data, list) else []
    selected_rows_list = selected_rows if isinstance(selected_rows, list) else []

    target_date = _coerce_date(picker_date) or us_eastern_today()
    target_date_iso = target_date.isoformat()

    initial_logs = log_state if isinstance(log_state, list) else []
    log_entries = initial_logs

    def _emit(message: str) -> None:
        nonlocal log_entries
        if not message:
            return
        log_entries = append_log(log_entries, message)

    cached_entry = _get_cached_earnings(target_date)
    cached_rows: list[dict[str, T.Any]] = []
    cached_options_flag: bool | None = None
    if isinstance(cached_entry, dict):
        payload = cached_entry.get("rowData")
        if isinstance(payload, list):
            cached_rows = payload
        cached_options_flag = cached_entry.get("options_filter_applied")

    effective_rows = table_rows if table_rows else cached_rows

    needs_refresh = False
    if not effective_rows:
        needs_refresh = True
    elif cached_options_flag is False and session_data:
        needs_refresh = True

    if needs_refresh:
        _emit(f"未找到已筛选的财报列表，自动刷新 {target_date} 的数据。")
        fetched_rows, status_text, _, options_applied, error_message = _prepare_earnings_dataset(
            target_date,
            session_data,
            username,
            password,
            twofa,
            logger=_emit,
        )
        if error_message:
            failure_detail = f"自动获取财报列表失败：{error_message}"
            _emit(failure_detail)
            failure_updates = []
            for cfg in PREDICTION_TIMELINES:
                task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
                failure_updates.append(
                    {
                        "id": task_id,
                        "name": name,
                        "status": "失败",
                        "detail": f"财报列表异常，跳过 {offset_label}",
                    }
                )
            task_state_local = _merge_task_updates(task_state_local, failure_updates, target_date=target_date_iso)
            tasks_changed = True
            log_output = log_entries if log_entries != initial_logs else no_update
            return (
                {"results": [], "missing": [], "errors": [], "rl_snapshot": initial_snapshot},
                failure_detail,
                initial_snapshot,
                task_state_local,
                log_output,
            )
        effective_rows = fetched_rows
        cached_options_flag = options_applied
        if status_text:
            _emit(status_text)
        _store_cached_earnings(
            target_date,
            fetched_rows,
            status_text or "",
            options_filter_applied=cached_options_flag,
        )
    elif use_all_rows and table_rows:
        _emit(f"使用当前表格中的财报列表，共 {len(table_rows)} 个标的。")

    metadata_map: dict[str, dict[str, T.Any]] = {}
    for row in effective_rows:
        if isinstance(row, dict) and row.get("symbol"):
            metadata_map[str(row["symbol"]).upper()] = row

    symbols: list[str] = []
    source_rows = effective_rows if use_all_rows else selected_rows_list
    for row in source_rows:
        if isinstance(row, dict) and row.get("symbol"):
            symbol = str(row["symbol"])
            symbols.append(symbol)
            metadata_map.setdefault(symbol.upper(), row)

    if use_all_rows and not symbols and effective_rows:
        for row in effective_rows:
            if isinstance(row, dict) and row.get("symbol"):
                symbol = str(row["symbol"])
                symbols.append(symbol)
                metadata_map.setdefault(symbol.upper(), row)

    unique_symbols = list(dict.fromkeys([s.upper() for s in symbols if s]))
    if not unique_symbols:
        message = "自动任务尚未选择标的，请等待财报列表加载。" if use_all_rows else "未选择标的。"
        if use_all_rows and not effective_rows:
            failure_updates = []
            for cfg in PREDICTION_TIMELINES:
                task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
                failure_updates.append(
                    {
                        "id": task_id,
                        "name": name,
                        "status": "失败",
                        "detail": f"未获取到财报列表，跳过 {offset_label}",
                    }
                )
            task_state_local = _merge_task_updates(task_state_local, failure_updates, target_date=target_date_iso)
            tasks_changed = True
        log_output = log_entries if log_entries != initial_logs else no_update
        task_output = task_state_local if tasks_changed else no_update
        return (
            {"results": [], "missing": [], "errors": [], "rl_snapshot": initial_snapshot},
            message,
            initial_snapshot,
            task_output,
            log_output,
        )

    timeline_summary: dict[str, dict[str, T.Any]] = {
        str(cfg.get("key")): {"label": cfg.get("label"), "success": 0, "missing": 0, "error": 0}
        for cfg in PREDICTION_TIMELINES
    }

    all_results: list[dict[str, T.Any]] = []
    all_missing: list[str] = []
    all_errors: list[str] = []
    timeline_reports: dict[str, str] = {}
    task_lookup: dict[str, tuple[str, str, str]] = {}
    progress_tracker: dict[str, dict[str, int]] = {}

    def _progress(symbol: str, timeline: dict[str, T.Any], stage: str, detail: str | None) -> None:
        nonlocal log_entries, task_state_local, tasks_changed

        label = str(timeline.get("label") or timeline.get("key") or "未知时点")
        key = str(timeline.get("key") or label)
        message: str | None = None
        if stage == "start":
            message = f"开始生成 {symbol} 的 {label} 预测。"
        elif stage == "success":
            timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
            timeline_summary[key]["success"] = timeline_summary[key].get("success", 0) + 1
            message = f"完成 {symbol} 的 {label} 预测。"
        elif stage == "missing":
            timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
            timeline_summary[key]["missing"] = timeline_summary[key].get("missing", 0) + 1
            message = f"{symbol} 的 {label} 输入缺失，跳过该时点。"
        elif stage == "error":
            timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
            timeline_summary[key]["error"] = timeline_summary[key].get("error", 0) + 1
            detail_text = f"：{detail}" if detail else ""
            message = f"{symbol} 的 {label} 预测失败{detail_text}。"
        if message:
            _emit(message)

        if stage in {"success", "missing", "error"}:
            tracker = progress_tracker.setdefault(
                key,
                {"total": len(unique_symbols), "completed": 0, "processed": 0},
            )
            if stage == "success":
                tracker["completed"] = tracker.get("completed", 0) + 1
            tracker["processed"] = tracker.get("processed", 0) + 1

            lookup = task_lookup.get(key)
            if lookup:
                progress_parts = [
                    f"共 {tracker['total']} 个标的",
                    f"已完成 {tracker['completed']} 个",
                ]
                if tracker["processed"] > tracker["completed"]:
                    progress_parts.append(f"已处理 {tracker['processed']} 个")
                if stage == "error" and detail:
                    progress_parts.append(f"最近错误：{detail}")
                update_payload = {
                    "id": lookup[0],
                    "name": lookup[1],
                    "detail": f"{lookup[2]}：" + "，".join(progress_parts),
                    "total_symbols": tracker["total"],
                    "completed_symbols": tracker["completed"],
                    "processed_symbols": tracker["processed"],
                }
                task_state_local = _merge_task_updates(
                    task_state_local,
                    [update_payload],
                    target_date=target_date_iso,
                )
                tasks_changed = True

    for cfg in PREDICTION_TIMELINES:
        task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
        start_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
        timeline_key = str(cfg.get("key") or offset_label)
        task_lookup[timeline_key] = (task_id, name, offset_label)
        progress_tracker.setdefault(
            timeline_key,
            {"total": len(unique_symbols), "completed": 0, "processed": 0},
        )
        task_state_local = _merge_task_updates(
            task_state_local,
            [
                {
                    "id": task_id,
                    "name": name,
                    "status": "进行中",
                    "detail": f"正在处理 {len(unique_symbols)} 个标的 ({offset_label})",
                    "start_time": start_ts,
                    "end_time": "",
                    "total_symbols": len(unique_symbols),
                    "completed_symbols": 0,
                    "processed_symbols": 0,
                }
            ],
            target_date=target_date_iso,
        )
        tasks_changed = True
        _emit(f"{name} 已开始。")

        partial_results, partial_missing, partial_errors = _compute_dci_for_symbols(
            unique_symbols,
            metadata_map,
            progress_callback=_progress,
            timeline_configs=[cfg],
        )

        all_results.extend(partial_results)
        all_missing.extend(partial_missing)
        all_errors.extend(partial_errors)

        key = str(cfg.get("key"))
        summary = timeline_summary.get(key) or {}
        success = int(summary.get("success", 0) or 0)
        missing_cnt = int(summary.get("missing", 0) or 0)
        error_cnt = int(summary.get("error", 0) or 0)
        parts: list[str] = []
        if success:
            parts.append(f"完成{success}次")
        if missing_cnt:
            parts.append(f"缺数据{missing_cnt}次")
        if error_cnt:
            parts.append(f"失败{error_cnt}次")
        detail_text = f"{offset_label}：" + ("，".join(parts) if parts else "无数据")
        tracker_final = progress_tracker.get(timeline_key)
        total_symbols_for_task = (
            tracker_final.get("total")
            if isinstance(tracker_final, dict)
            else len(unique_symbols)
        )
        completed_for_task = (
            tracker_final.get("completed")
            if isinstance(tracker_final, dict)
            else success
        )
        processed_for_task = (
            tracker_final.get("processed")
            if isinstance(tracker_final, dict)
            else success + missing_cnt + error_cnt
        )
        if total_symbols_for_task > 0:
            progress_phrase = f"{completed_for_task}/{total_symbols_for_task}"
        else:
            progress_phrase = str(completed_for_task)
        progress_suffix = (
            f"，已处理 {processed_for_task} 个"
            if processed_for_task > completed_for_task
            else ""
        )
        detail_text = f"{detail_text}｜进度：{progress_phrase}{progress_suffix}"
        timeline_reports[key] = detail_text

        if success:
            status_value = "已完成"
        elif error_cnt:
            status_value = "失败"
        else:
            status_value = "无数据"

        end_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
        task_state_local = _merge_task_updates(
            task_state_local,
            [
                {
                    "id": task_id,
                    "name": name,
                    "status": status_value,
                    "detail": detail_text,
                    "end_time": end_ts,
                    "total_symbols": total_symbols_for_task,
                    "completed_symbols": completed_for_task,
                    "processed_symbols": processed_for_task,
                }
            ],
            target_date=target_date_iso,
        )
        tasks_changed = True
        _emit(f"{name} 完成：{detail_text}")

    results = sorted(all_results, key=lambda row: (row.get("symbol", ""), row.get("lookback_days", 0)))
    missing = all_missing
    errors = all_errors
    snapshot = RL_MANAGER.snapshot() if RL_MANAGER is not None else None

    lines = []
    if results:
        lines.append(f"已生成 {len(results)} 个预测结果（每个标的最多 {len(PREDICTION_TIMELINES)} 个时点）。")
    else:
        lines.append("未生成有效预测，请检查 DCI 输入数据。")

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        lines.append(f"缺少 DCI 输入：{unique_missing}。")
    if errors:
        lines.append("计算时出现错误：" + "; ".join(errors))

    if timeline_reports:
        lines.append("各时点统计：")
        for cfg in PREDICTION_TIMELINES:
            key = str(cfg.get("key"))
            report = timeline_reports.get(key)
            if report:
                label = cfg.get("label") or cfg.get("key")
                lines.append(f"- {label}：{report.split('：', 1)[-1]}")

    message = "\n".join(lines)

    if timeline_reports:
        ordered_reports = [timeline_reports.get(str(cfg.get("key"))) for cfg in PREDICTION_TIMELINES]
        summary_texts = [
            rep.replace("：", " -> ", 1)
            for rep in ordered_reports
            if rep
        ]
        if summary_texts:
            _emit("时点统计：" + "；".join(summary_texts))

    log_output = log_entries if log_entries != initial_logs else no_update

    task_updates = task_state_local if tasks_changed else no_update

    stored_row_data = effective_rows if isinstance(effective_rows, list) else []
    _store_prediction_results(target_date, stored_row_data, results, message or "")

    archive_entry = _get_prediction_archive(target_date)
    if isinstance(archive_entry, dict):
        evaluation = archive_entry.get("evaluation")
        if isinstance(evaluation, dict):
            actual_map: dict[str, dict[str, T.Any]] = {}
            items = evaluation.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    key = f"{item.get('symbol', '')}::{item.get('timeline_key', '')}"
                    actual_map[key] = item
            if actual_map:
                for entry in results:
                    if not isinstance(entry, dict):
                        continue
                    key = f"{entry.get('symbol', '')}::{entry.get('timeline_key', '')}"
                    actual = actual_map.get(key)
                    if not actual:
                        continue
                    entry["actual_direction"] = actual.get("actual_direction")
                    entry["actual_move_pct"] = actual.get("actual_move_pct")
                    entry["prediction_correct"] = actual.get("prediction_correct")
                    entry["actual_checked_at"] = evaluation.get("checked_at")

    return (
        {"results": results, "missing": missing, "errors": errors, "rl_snapshot": snapshot},
        message or "",
        snapshot,
        task_updates,
        log_output,
    )


@app.callback(
    Output("prediction-table", "rowData"),
    Input("prediction-store", "data"),
)
def render_prediction_table(store_data):  # noqa: D401
    if not isinstance(store_data, dict):
        return []
    results = store_data.get("results")
    if isinstance(results, list):
        pending: list[dict[str, T.Any]] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            if row.get("actual_direction"):
                continue
            pending.append(row)
        return pending
    return []


@app.callback(
    Output("evaluation-store", "data"),
    Output("rl-agent-store", "data", allow_duplicate=True),
    Input("post-open-eval", "n_intervals"),
    State("evaluation-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def auto_evaluate_predictions(n_intervals, existing_store):  # noqa: D401
    del n_intervals

    now = dt.datetime.now(US_EASTERN)
    if now.hour < 9 or (now.hour == 9 and now.minute < 40):
        return no_update, no_update

    today = now.date()
    today_iso = today.isoformat()

    store = existing_store if isinstance(existing_store, dict) else {}
    if store.get("last_run") == today_iso:
        return no_update, no_update

    target_date = previous_trading_day(today)
    archive_entry = _get_prediction_archive(target_date)

    evaluation_items: list[dict[str, T.Any]] = []
    errors: list[str] = []
    correct_count = 0
    move_samples: list[float] = []

    results = []
    if isinstance(archive_entry, dict):
        raw_results = archive_entry.get("results")
        if isinstance(raw_results, list):
            results = [
                r
                for r in raw_results
                if isinstance(r, dict)
                and (str(r.get("timeline_key")) == "decision_day" or int(r.get("lookback_days", 0)) == 0)
            ]

    if not results:
        evaluation_payload = {
            "date": target_date.isoformat(),
            "checked_at": now.isoformat(),
            "items": [],
            "summary": {
                "date": target_date.isoformat(),
                "checked_at": now.isoformat(),
                "total": 0,
                "correct": 0,
                "success_rate": 0.0,
                "avg_move": None,
            },
            "message": "昨日未记录可检验的预测。",
        }
        _store_prediction_evaluation(target_date, evaluation_payload)
        evaluation_store = {
            "last_run": today_iso,
            "date": target_date.isoformat(),
            "checked_at": now.isoformat(),
            "items": [],
            "summary": evaluation_payload["summary"],
            "message": evaluation_payload.get("message"),
        }
        return evaluation_store, no_update

    for entry in results:
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": FINNHUB_API_KEY},
                timeout=10,
            )
            data = resp.json() if resp.ok else None
        except (requests.RequestException, ValueError) as exc:
            data = None
            errors.append(f"{symbol}: {exc}")

        actual_direction = "数据缺失"
        move_pct: float | None = None
        correct_text = "-"
        actual_up: bool | None = None

        if isinstance(data, dict):
            current_price = float(data.get("c") or 0.0)
            prev_close = float(data.get("pc") or 0.0)
            if prev_close > 0:
                move_pct = round(((current_price - prev_close) / prev_close) * 100.0, 2)
                move_samples.append(abs(move_pct))
                if move_pct > 0:
                    actual_direction = "上涨"
                    actual_up = True
                elif move_pct < 0:
                    actual_direction = "下跌"
                    actual_up = False
                else:
                    actual_direction = "持平"
                    actual_up = True
                predicted = str(entry.get("direction") or "")
                if actual_direction == "上涨":
                    is_correct = predicted == "买入"
                elif actual_direction == "下跌":
                    is_correct = predicted == "卖出"
                else:
                    is_correct = True
                correct_text = "是" if is_correct else "否"
                if is_correct:
                    correct_count += 1
            else:
                errors.append(f"{symbol}: 缺少上一交易日收盘价")
        else:
            errors.append(f"{symbol}: 获取行情失败")

        item = {
            "symbol": symbol,
            "company": entry.get("company"),
            "sector": entry.get("sector"),
            "predicted_direction": entry.get("direction"),
            "actual_direction": actual_direction,
            "actual_move_pct": move_pct,
            "prediction_correct": correct_text,
            "timeline_label": entry.get("timeline_label"),
            "timeline_key": entry.get("timeline_key"),
            "checked_at": now.isoformat(),
        }
        evaluation_items.append(item)

        if RL_MANAGER is not None and actual_up is not None:
            try:
                RL_MANAGER.apply_feedback(
                    symbol,
                    actual_up=actual_up,
                    actual_move_pct=move_pct,
                    prediction_id=entry.get("rl_prediction_id"),
                    sector=entry.get("sector") if entry.get("sector") not in (None, "未知") else None,
                )
            except Exception:
                pass

    total = len(evaluation_items)
    success_rate = (correct_count / total) if total else 0.0
    avg_move = sum(move_samples) / len(move_samples) if move_samples else None

    summary = {
        "date": target_date.isoformat(),
        "checked_at": now.isoformat(),
        "total": total,
        "correct": correct_count,
        "success_rate": success_rate,
        "avg_move": avg_move,
        "errors": errors,
    }

    evaluation_payload = {
        "date": target_date.isoformat(),
        "checked_at": now.isoformat(),
        "items": evaluation_items,
        "summary": summary,
    }
    if errors:
        evaluation_payload["errors"] = errors

    _store_prediction_evaluation(target_date, evaluation_payload)

    evaluation_store = {
        "last_run": today_iso,
        "date": target_date.isoformat(),
        "checked_at": now.isoformat(),
        "items": evaluation_items,
        "summary": summary,
    }
    if errors:
        evaluation_store["message"] = "; ".join(errors)

    snapshot = RL_MANAGER.snapshot() if RL_MANAGER is not None else None
    return evaluation_store, snapshot or no_update


@app.callback(
    Output("prediction-store", "data", allow_duplicate=True),
    Input("evaluation-store", "data"),
    State("prediction-store", "data"),
    prevent_initial_call=True,
)
def sync_actual_into_predictions(evaluation_store, prediction_store):  # noqa: D401
    if not isinstance(evaluation_store, dict):
        return no_update

    items = evaluation_store.get("items")
    if not isinstance(items, list) or not items:
        return no_update

    if not isinstance(prediction_store, dict):
        return no_update

    results = prediction_store.get("results")
    if not isinstance(results, list) or not results:
        return no_update

    actual_map: dict[tuple[str, str], dict[str, T.Any]] = {}
    symbol_map: dict[str, dict[str, T.Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        timeline = str(item.get("timeline_key") or "decision_day")
        actual_map[(symbol, timeline)] = item
        symbol_map.setdefault(symbol, item)

    if not actual_map and not symbol_map:
        return no_update

    checked_at = evaluation_store.get("checked_at")
    changed = False
    new_results: list[dict[str, T.Any]] = []

    for entry in results:
        if not isinstance(entry, dict):
            new_results.append(entry)
            continue

        symbol = str(entry.get("symbol") or "").upper()
        timeline = str(entry.get("timeline_key") or "decision_day")
        actual = actual_map.get((symbol, timeline)) or symbol_map.get(symbol)

        if actual:
            new_entry = copy.deepcopy(entry)
            actual_direction = actual.get("actual_direction")
            actual_move = actual.get("actual_move_pct")
            prediction_correct = actual.get("prediction_correct")
            checked_value = actual.get("checked_at") or checked_at

            if (
                new_entry.get("actual_direction") != actual_direction
                or new_entry.get("actual_move_pct") != actual_move
                or new_entry.get("prediction_correct") != prediction_correct
                or new_entry.get("actual_checked_at") != checked_value
            ):
                new_entry["actual_direction"] = actual_direction
                new_entry["actual_move_pct"] = actual_move
                new_entry["prediction_correct"] = prediction_correct
                new_entry["actual_checked_at"] = checked_value
                changed = True
            new_results.append(new_entry)
        else:
            new_results.append(entry)

    if not changed:
        return no_update

    new_store = copy.deepcopy(prediction_store)
    new_store["results"] = new_results
    return new_store


@app.callback(
    Output("task-table", "rowData"),
    Input("task-store", "data"),
)
def render_task_table(task_state):  # noqa: D401
    if not isinstance(task_state, dict):
        return []

    tasks = task_state.get("tasks")
    if not isinstance(tasks, list):
        return []

    return tasks


@app.callback(
    Output("model-global-summary", "children"),
    Output("model-sector-table", "rowData"),
    Input("rl-agent-store", "data"),
)
def render_model_parameters(agent_data):  # noqa: D401
    default_summary = "模型尚未初始化。"
    if not isinstance(agent_data, dict):
        return default_summary, []

    global_data = agent_data.get("global") if "global" in agent_data else agent_data
    sectors = agent_data.get("sectors") if isinstance(agent_data.get("sectors"), dict) else {}

    lines: list[str] = []
    if isinstance(global_data, dict):
        lines.append(f"学习率: {global_data.get('learning_rate')}")
        lines.append(f"折现因子: {global_data.get('gamma')}")
        lines.append(f"调整系数: {global_data.get('adjustment_scale')}")
        lines.append(f"偏置: {round(float(global_data.get('bias', 0.0)), 4)}")
        lines.append(f"当前基准: {round(float(global_data.get('baseline', 0.0)), 4)}")
        lines.append(f"累计更新: {global_data.get('update_count')}")
        weights = global_data.get("weights")
        if isinstance(weights, dict) and weights:
            sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            top_weights = sorted_weights[:10]
            lines.append("主要权重:")
            for key, value in top_weights:
                lines.append(f"  {key}: {round(float(value), 5)}")
        else:
            lines.append("主要权重: 无")
        factor_weights = agent_data.get("factor_weights")
        if isinstance(factor_weights, dict) and factor_weights:
            lines.append("当前因子权重:")
            for key, value in sorted(factor_weights.items(), key=lambda kv: kv[0]):
                lines.append(f"  {key}: {round(float(value), 4)}")
    else:
        lines.append(default_summary)

    sector_rows: list[dict[str, T.Any]] = []
    if sectors:
        for sector_name, payload in sectors.items():
            if not isinstance(payload, dict):
                continue
            weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
            top_display = ""
            if weights:
                sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
                top_display = "; ".join(
                    f"{k}:{round(float(v), 4)}" for k, v in sorted_weights[:5]
                )
            sector_rows.append(
                {
                    "sector": sector_name,
                    "total_predictions": payload.get("total_predictions") or payload.get("prediction_count"),
                    "update_count": payload.get("update_count"),
                    "baseline": round(float(payload.get("baseline", 0.0)), 4),
                    "top_weights": top_display or "-",
                }
            )

    return "\n".join(lines), sector_rows


@app.callback(
    Output("rl-model-table", "rowData"),
    Input("rl-agent-store", "data"),
)
def render_model_details(agent_data):  # noqa: D401
    if not isinstance(agent_data, dict):
        return []

    rows: list[dict[str, T.Any]] = []

    def _format_value(value: T.Any) -> T.Any:
        if isinstance(value, float):
            return round(float(value), 6)
        return value

    global_data = agent_data.get("global") if "global" in agent_data else agent_data
    if isinstance(global_data, dict):
        for key in ("learning_rate", "gamma", "adjustment_scale", "bias", "baseline", "update_count"):
            if key in global_data:
                rows.append(
                    {
                        "model": "全局",
                        "parameter": key,
                        "value": _format_value(global_data.get(key)),
                        "description": RL_PARAM_DESCRIPTIONS.get(key, "参数说明待补充。"),
                    }
                )
        weights = global_data.get("weights") if isinstance(global_data.get("weights"), dict) else {}
        if weights:
            sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            for feature, weight in sorted_weights[:10]:
                rows.append(
                    {
                        "model": "全局-权重",
                        "parameter": feature,
                        "value": _format_value(weight),
                        "description": RL_PARAM_DESCRIPTIONS.get("weights", "特征权重。"),
                    }
                )

        factor_weights = agent_data.get("factor_weights")
        if isinstance(factor_weights, dict) and factor_weights:
            for factor_name, weight in sorted(factor_weights.items(), key=lambda kv: kv[0]):
                rows.append(
                    {
                        "model": "DCI-因子权重",
                        "parameter": factor_name,
                        "value": _format_value(weight),
                        "description": FACTOR_DESCRIPTIONS.get(
                            factor_name,
                            RL_PARAM_DESCRIPTIONS.get("factor_weights", "DCI 因子权重。"),
                        ),
                    }
                )

    sectors = agent_data.get("sectors") if isinstance(agent_data.get("sectors"), dict) else {}
    if sectors:
        for sector_name, payload in sectors.items():
            if not isinstance(payload, dict):
                continue
            for key in ("baseline", "update_count", "total_predictions"):
                if key in payload:
                    rows.append(
                        {
                            "model": f"行业:{sector_name}",
                            "parameter": key,
                            "value": _format_value(payload.get(key)),
                            "description": RL_PARAM_DESCRIPTIONS.get(key, "参数说明待补充。"),
                        }
                    )
            weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
            if weights:
                sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
                for feature, weight in sorted_weights[:5]:
                    rows.append(
                        {
                            "model": f"行业:{sector_name}-权重",
                            "parameter": feature,
                            "value": _format_value(weight),
                            "description": RL_PARAM_DESCRIPTIONS.get("weights", "特征权重。"),
                        }
                    )

    return rows


def _normalise_history_value(value: T.Any) -> T.Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 6)
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(_normalise_history_value(item)) for item in value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _extract_global_trackable(agent_data: dict[str, T.Any]) -> dict[str, T.Any]:
    result: dict[str, T.Any] = {}
    global_data = agent_data.get("global") if isinstance(agent_data.get("global"), dict) else agent_data
    if isinstance(global_data, dict):
        for key in ("learning_rate", "gamma", "adjustment_scale", "bias", "baseline", "update_count", "total_predictions"):
            if key in global_data:
                result[key] = global_data.get(key)
        weights = global_data.get("weights") if isinstance(global_data.get("weights"), dict) else {}
        if weights:
            sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            top_display = "; ".join(
                f"{name}:{round(float(value), 4)}" for name, value in sorted_weights[:5]
            )
            result["weights:top"] = top_display
    factor_weights = agent_data.get("factor_weights") if isinstance(agent_data.get("factor_weights"), dict) else {}
    for factor_name, value in sorted(factor_weights.items(), key=lambda kv: kv[0]):
        result[f"factor_weights::{factor_name}"] = value
    return result


def _extract_sector_trackable(agent_data: dict[str, T.Any]) -> dict[str, dict[str, T.Any]]:
    sectors_map: dict[str, dict[str, T.Any]] = {}
    sectors = agent_data.get("sectors") if isinstance(agent_data.get("sectors"), dict) else {}
    for sector_name, payload in sectors.items():
        if not isinstance(payload, dict):
            continue
        params: dict[str, T.Any] = {}
        for key in ("baseline", "update_count", "total_predictions"):
            if key in payload:
                params[key] = payload.get(key)
        weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
        if weights:
            sorted_weights = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            params["weights:top"] = "; ".join(
                f"{name}:{round(float(value), 4)}" for name, value in sorted_weights[:5]
            )
        if params:
            sectors_map[sector_name] = params
    return sectors_map


@app.callback(
    Output("rl-parameter-history", "data"),
    Input("rl-agent-store", "data"),
    State("rl-parameter-history", "data"),
)
def update_parameter_history(agent_data, history_state):  # noqa: D401
    base_history: dict[str, T.Any]
    if isinstance(history_state, dict):
        base_history = history_state
    else:
        base_history = {"global": {"last": {}, "changes": []}, "sectors": {}}

    if not isinstance(agent_data, dict):
        return base_history

    timestamp = dt.datetime.now(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S")
    history = copy.deepcopy(base_history)
    max_records = 200
    recorded_global_changes: list[dict[str, T.Any]] = []
    recorded_sector_changes: list[dict[str, T.Any]] = []

    global_section = history.setdefault("global", {"last": {}, "changes": []})
    previous_global = global_section.get("last") if isinstance(global_section.get("last"), dict) else {}
    current_global = _extract_global_trackable(agent_data)
    global_changes: list[dict[str, T.Any]] = []
    for key, value in current_global.items():
        norm_value = _normalise_history_value(value)
        norm_previous = _normalise_history_value(previous_global.get(key))
        if norm_value != norm_previous:
            global_changes.append(
                {
                    "timestamp": timestamp,
                    "parameter": key,
                    "old_value": norm_previous,
                    "new_value": norm_value,
                }
            )
    for key in previous_global:
        if key not in current_global:
            norm_previous = _normalise_history_value(previous_global.get(key))
            global_changes.append(
                {
                    "timestamp": timestamp,
                    "parameter": key,
                    "old_value": norm_previous,
                    "new_value": None,
                }
            )
    if global_changes:
        existing_changes = global_section.get("changes") if isinstance(global_section.get("changes"), list) else []
        global_section["changes"] = (existing_changes + global_changes)[-max_records:]
        recorded_global_changes.extend(copy.deepcopy(global_changes))
    global_section["last"] = {k: _normalise_history_value(v) for k, v in current_global.items()}

    sectors_section = history.setdefault("sectors", {})
    current_sectors = _extract_sector_trackable(agent_data)
    for sector_name, params in current_sectors.items():
        sector_entry = sectors_section.setdefault(sector_name, {"last": {}, "changes": []})
        previous_params = sector_entry.get("last") if isinstance(sector_entry.get("last"), dict) else {}
        sector_changes: list[dict[str, T.Any]] = []
        for key, value in params.items():
            norm_value = _normalise_history_value(value)
            norm_previous = _normalise_history_value(previous_params.get(key))
            if norm_value != norm_previous:
                sector_changes.append(
                    {
                        "timestamp": timestamp,
                        "sector": sector_name,
                        "parameter": key,
                        "old_value": norm_previous,
                        "new_value": norm_value,
                    }
                )
        for key in previous_params:
            if key not in params:
                norm_previous = _normalise_history_value(previous_params.get(key))
                sector_changes.append(
                    {
                        "timestamp": timestamp,
                        "sector": sector_name,
                        "parameter": key,
                        "old_value": norm_previous,
                        "new_value": None,
                    }
                )
        if sector_changes:
            existing_sector_changes = sector_entry.get("changes") if isinstance(sector_entry.get("changes"), list) else []
            sector_entry["changes"] = (existing_sector_changes + sector_changes)[-max_records:]
            recorded_sector_changes.extend(copy.deepcopy(sector_changes))
        sector_entry["last"] = {k: _normalise_history_value(v) for k, v in params.items()}

    removed_sectors = [name for name in sectors_section.keys() if name not in current_sectors]
    for sector_name in removed_sectors:
        entry = sectors_section.get(sector_name)
        if not isinstance(entry, dict):
            continue
        previous_params = entry.get("last") if isinstance(entry.get("last"), dict) else {}
        if previous_params:
            removal_entries = [
                {
                    "timestamp": timestamp,
                    "sector": sector_name,
                    "parameter": key,
                    "old_value": _normalise_history_value(value),
                    "new_value": None,
                }
                for key, value in previous_params.items()
            ]
            existing_changes = entry.get("changes") if isinstance(entry.get("changes"), list) else []
            entry["changes"] = (existing_changes + removal_entries)[-max_records:]
            recorded_sector_changes.extend(copy.deepcopy(removal_entries))
        entry["last"] = {}

    if recorded_global_changes or recorded_sector_changes:
        snapshot_payload = {
            "timestamp": timestamp,
            "global_changes": recorded_global_changes,
            "sector_changes": recorded_sector_changes,
            "state": copy.deepcopy(history),
        }
        _persist_parameter_snapshot(timestamp, snapshot_payload)

    return history


@app.callback(
    Output("model-global-history-table", "rowData"),
    Output("model-sector-history-table", "rowData"),
    Input("rl-parameter-history", "data"),
)
def render_parameter_history_tables(history_state):  # noqa: D401
    if not isinstance(history_state, dict):
        return [], []

    global_rows: list[dict[str, T.Any]] = []
    global_section = history_state.get("global") if isinstance(history_state.get("global"), dict) else {}
    global_changes = global_section.get("changes") if isinstance(global_section.get("changes"), list) else []
    for change in global_changes:
        if not isinstance(change, dict):
            continue
        global_rows.append(
            {
                "timestamp": change.get("timestamp"),
                "parameter": change.get("parameter"),
                "old_value": change.get("old_value"),
                "new_value": change.get("new_value"),
            }
        )
    global_rows = sorted(global_rows, key=lambda row: row.get("timestamp") or "", reverse=True)

    sector_rows: list[dict[str, T.Any]] = []
    sectors_section = history_state.get("sectors") if isinstance(history_state.get("sectors"), dict) else {}
    for sector_name, payload in sectors_section.items():
        if not isinstance(payload, dict):
            continue
        changes = payload.get("changes") if isinstance(payload.get("changes"), list) else []
        for change in changes:
            if not isinstance(change, dict):
                continue
            sector_rows.append(
                {
                    "timestamp": change.get("timestamp"),
                    "sector": change.get("sector") or sector_name,
                    "parameter": change.get("parameter"),
                    "old_value": change.get("old_value"),
                    "new_value": change.get("new_value"),
                }
            )
    sector_rows = sorted(sector_rows, key=lambda row: row.get("timestamp") or "", reverse=True)

    return global_rows, sector_rows


@app.callback(
    Output("log-output", "children"),
    Input("log-store", "data"),
)
def update_log_output(log_state):
    if not isinstance(log_state, list) or not log_state:
        return "暂无日志记录。"
    return "\n".join(log_state)


@app.callback(
    Output("validation-date-dropdown", "options"),
    Output("validation-date-dropdown", "value"),
    Output("validation-symbol-dropdown", "options"),
    Output("validation-symbol-dropdown", "value"),
    Output("validation-status", "children"),
    Output("validation-table", "rowData"),
    Output("validation-graph-dci", "figure"),
    Output("validation-graph-prob", "figure"),
    Input("validation-date-dropdown", "value"),
    Input("validation-symbol-dropdown", "value"),
    Input("evaluation-store", "data"),
    Input("prediction-store", "data"),
)
def render_validation_view(selected_date, selected_symbol, evaluation_store, prediction_store):  # noqa: D401
    del prediction_store

    archive = _load_prediction_archive_raw()
    evaluation_override: dict[str, dict[str, T.Any]] = {}
    if isinstance(evaluation_store, dict) and evaluation_store.get("date"):
        evaluation_override[str(evaluation_store.get("date"))] = evaluation_store

    available_entries: list[tuple[str, dict[str, T.Any], dict[str, T.Any]]] = []
    for date_key, entry in sorted(archive.items(), key=lambda item: item[0], reverse=True):
        if not isinstance(entry, dict):
            continue
        results = entry.get("results")
        if not isinstance(results, list) or not results:
            continue
        evaluation = evaluation_override.get(date_key) or entry.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        items = evaluation.get("items")
        if not isinstance(items, list) or not items:
            continue
        available_entries.append((date_key, entry, evaluation))

    date_options = [
        {"label": date_key, "value": date_key}
        for date_key, _, _ in available_entries
    ]

    if not date_options:
        message = "暂未找到已检验的预测记录。"
        return (
            [],
            None,
            [],
            None,
            message,
            [],
            _empty_figure("暂无数据"),
            _empty_figure("暂无数据"),
        )

    date_values = {opt["value"] for opt in date_options}
    current_date = selected_date if selected_date in date_values else date_options[0]["value"]

    entry_map = {date_key: (entry, evaluation) for date_key, entry, evaluation in available_entries}
    entry, evaluation = entry_map[current_date]
    results = entry.get("results") if isinstance(entry.get("results"), list) else []
    evaluation_items = evaluation.get("items") if isinstance(evaluation.get("items"), list) else []

    symbol_options: list[dict[str, str]] = []
    symbol_set: dict[str, dict[str, T.Any]] = {}
    for item in evaluation_items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if not symbol or symbol in symbol_set:
            continue
        symbol_set[symbol] = item
        label = f"{symbol}"
        company = item.get("company")
        if company:
            label = f"{symbol} - {company}"
        symbol_options.append({"label": label, "value": symbol})

    if not symbol_options:
        message = "所选日期没有对应的检验记录。"
        return (
            date_options,
            current_date,
            [],
            None,
            message,
            [],
            _empty_figure("暂无数据"),
            _empty_figure("暂无数据"),
        )

    valid_symbol_values = {opt["value"] for opt in symbol_options}
    current_symbol = selected_symbol if selected_symbol in valid_symbol_values else symbol_options[0]["value"]

    timeline_map: dict[str, dict[str, T.Any]] = {}
    for entry_row in results:
        if not isinstance(entry_row, dict):
            continue
        symbol = str(entry_row.get("symbol") or "").upper()
        if symbol != current_symbol:
            continue
        timeline_key = str(entry_row.get("timeline_key") or entry_row.get("timeline_alias") or "")
        if not timeline_key:
            continue
        timeline_map[timeline_key] = entry_row

    actual_entry = symbol_set.get(current_symbol)

    table_rows = _build_validation_rows(timeline_map, actual_entry)
    fig_dci, fig_prob = _build_validation_figures(timeline_map, actual_entry)

    if isinstance(actual_entry, dict):
        actual_direction = actual_entry.get("actual_direction", "未知")
        move_pct = actual_entry.get("actual_move_pct")
        if move_pct is None:
            move_text = "-"
        else:
            try:
                move_value = float(move_pct)
            except (TypeError, ValueError):
                move_text = f"{move_pct}%"
            else:
                move_text = f"{move_value:.2f}%"
        checked_at = actual_entry.get("checked_at") or evaluation.get("checked_at")
        status_message = (
            f"{current_symbol} | 实际方向：{actual_direction} | 实际涨跌幅：{move_text}"
        )
        if checked_at:
            status_message += f"\n检验时间：{checked_at}"
    else:
        status_message = "尚未获取该标的的实际结果。"

    return (
        date_options,
        current_date,
        symbol_options,
        current_symbol,
        status_message,
        table_rows,
        fig_dci,
        fig_prob,
    )


@app.callback(
    Output("overview-total-success", "figure"),
    Output("overview-sector-success", "figure"),
    Output("overview-timeline-trend", "figure"),
    Input("prediction-store", "data"),
    Input("evaluation-store", "data"),
)
def render_overview_charts(prediction_store, evaluation_store):  # noqa: D401
    del prediction_store, evaluation_store
    archive = _load_prediction_archive_raw()
    return build_overview_figures(archive)


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


@app.callback(
    Output("log-modal", "is_open"),
    Input("show-log-btn", "n_clicks"),
    Input("close-log-btn", "n_clicks"),
    State("log-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_log_modal(show_clicks, close_clicks, is_open):
    trigger = ctx.triggered_id
    if trigger == "show-log-btn":
        return True
    if trigger == "close-log-btn":
        return False
    return is_open


@app.callback(
    Output("dl", "data"),
    Input("dl-btn", "n_clicks"),
    State("table", "rowData"),
    prevent_initial_call=True,
)
def download_csv(n, data):
    del n
    if not data:
        return dcc.send_string(
            "\ufeffsymbol,company,bucket,decision_date\n",
            filename="earnings_weeklies.csv",
        )
    df = pd.DataFrame(data)
    if "weekly_exp_this_fri" in df.columns:
        df = df.drop(columns=["weekly_exp_this_fri"])
    return dcc.send_data_frame(
        df.to_csv,
        "earnings_weeklies.csv",
        index=False,
        encoding="utf-8-sig",
    )


def create_dash_app() -> Dash:
    """Return the configured Dash app instance."""

    return app

