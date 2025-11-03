import os
import json
import time
import threading
import typing as T
import datetime as dt
import uuid
import copy
import math
import traceback
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlparse

import requests
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html, ctx, no_update

from .dci import BASE_FACTOR_WEIGHTS, build_inputs, compute_dci, get_factor_weights
from .dci.providers import load_dci_payloads
from .data import finnhub, fred, nasdaq, openfigi, tv as tv_data

try:
    from .dci.rl import RLAgentManager, get_global_manager
except Exception:  # pragma: no cover - RL module is optional
    RL_MANAGER: "RLAgentManager | None" = None
else:
    RL_MANAGER = get_global_manager()

# ---------- Timezone helpers ----------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

US_EASTERN = ZoneInfo("America/New_York")

ARCHIVE_ROOT = Path(__file__).with_name("archives")
CACHE_ROOT = ARCHIVE_ROOT / "cache"

PREDICTION_LOCK = threading.Lock()
PREDICTION_ARCHIVE_DIR = ARCHIVE_ROOT / "predictions"
PREDICTION_ARCHIVE_PATH = PREDICTION_ARCHIVE_DIR / "index.json"

SYMBOL_META_LOCK = threading.Lock()
SYMBOL_META_PATH = CACHE_ROOT / "symbol_metadata.json"
SYMBOL_META_CACHE: dict[str, dict[str, T.Any]] = {}

LOG_ARCHIVE_DIR = ARCHIVE_ROOT / "logs"
LOG_LINES_PER_FILE = 2000
LOG_MAX_FILES = 20
LOG_FILE_LOCK = threading.Lock()
CURRENT_LOG_FILE: Path | None = None
CURRENT_LOG_LINE_COUNT = 0
PROCESS_ID = os.getpid()


def us_eastern_today() -> dt.date:
    """Return today's date in US Eastern timezone."""

    return dt.datetime.now(US_EASTERN).date()

TV_DEFAULT_EXCHANGE = os.getenv("TVDATAFEED_DEFAULT_EXCHANGE", "").upper().strip()
TV_OPTION_EXCHANGE = os.getenv("TVDATAFEED_OPTION_EXCHANGE", "CBOE").upper().strip()
TV_PRICE_TOLERANCE_MINUTES = max(
    1,
    int(os.getenv("TVDATAFEED_PRICE_TOLERANCE_MINUTES", "5") or 5),
)
TV_OPTION_TOLERANCE_MINUTES = max(
    1,
    int(os.getenv("TVDATAFEED_OPTION_TOLERANCE_MINUTES", "5") or 5),
)

TV_EXCHANGE_ALIASES = {
    "NASDAQ": "NASDAQ",
    "NASDAQ NMS": "NASDAQ",
    "NASDAQ GLOBAL SELECT": "NASDAQ",
    "NYSE": "NYSE",
    "NEW YORK STOCK EXCHANGE": "NYSE",
    "NYSE ARCA": "NYSEARCA",
    "ARCA": "NYSEARCA",
    "NYSEMKT": "AMEX",
    "AMEX": "AMEX",
    "BATS": "BATS",
}

PREDICTION_TIMELINES = [
    {
        "key": "plus14",
        "label": "决策日后14天",
        "lookback": 14,
        "offset_days": 14,
        "aliases": [
            "plus14",
            "d+14",
            "t+14",
            "p14",
            "minus14",
            "-14",
            "t-14",
            "d-14",
        ],
    },
    {
        "key": "plus7",
        "label": "决策日后7天",
        "lookback": 7,
        "offset_days": 7,
        "aliases": [
            "plus7",
            "d+7",
            "t+7",
            "p7",
            "minus7",
            "-7",
            "t-7",
            "d-7",
        ],
    },
    {
        "key": "plus3",
        "label": "决策日后3天",
        "lookback": 3,
        "offset_days": 3,
        "aliases": [
            "plus3",
            "d+3",
            "t+3",
            "p3",
            "minus3",
            "-3",
            "t-3",
            "d-3",
        ],
    },
    {
        "key": "plus1",
        "label": "决策日后1天",
        "lookback": 1,
        "offset_days": 1,
        "aliases": [
            "plus1",
            "d+1",
            "t+1",
            "p1",
            "minus1",
            "-1",
            "t-1",
            "d-1",
        ],
    },
    {
        "key": "decision_day",
        "label": "决策日收盘前",
        "lookback": 0,
        "offset_days": 0,
        "aliases": ["decision_day", "day0", "decision", "today", "0", "final"],
    },
]

DCI_AUTO_BASELINE_ENABLED = (
    str(os.getenv("DCI_AUTO_BASELINE", "1")).strip().lower() not in {"0", "false", "no"}
)

DAILY_T_MINUS_ONE_TASK_ID = "daily::t-1-eval"
DAILY_ADJUST_TASK_ID = "daily::adjust-params"
BACKFILL_TASK_ID = "backfill::validation"

EARNINGS_CACHE_PATH = CACHE_ROOT / "earnings.json"
EARNINGS_ARCHIVE_DIR = ARCHIVE_ROOT / "earnings"
CACHE_LOCK = threading.Lock()


def _load_archive_directory(directory: Path) -> dict[str, T.Any]:
    result: dict[str, T.Any] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.iterdir()):
        if path.is_file():
            if path.suffix.lower() != ".json" or path.name == "index.json":
                continue
            key = path.stem
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                result[key] = payload
            continue
        if path.is_dir():
            summary = path / "summary.json"
            if not summary.exists():
                continue
            try:
                with summary.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                result[path.name] = payload
    return result


def _write_archive_file(path: Path, payload: dict[str, T.Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _json_safe(value: T.Any) -> T.Any:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return value


def _reset_log_rotation_state() -> None:
    global CURRENT_LOG_FILE, CURRENT_LOG_LINE_COUNT
    CURRENT_LOG_FILE = None
    CURRENT_LOG_LINE_COUNT = 0


def _enforce_log_rotation() -> None:
    try:
        files = sorted(
            LOG_ARCHIVE_DIR.glob(f"log-{PROCESS_ID}-*.txt"),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return
    excess = len(files) - LOG_MAX_FILES
    for _ in range(max(excess, 0)):
        oldest = files.pop(0)
        try:
            oldest.unlink()
        except OSError:
            continue


def _persist_log_entry(entry: str) -> None:
    global CURRENT_LOG_FILE, CURRENT_LOG_LINE_COUNT
    try:
        with LOG_FILE_LOCK:
            LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            needs_new_file = (
                CURRENT_LOG_FILE is None
                or CURRENT_LOG_LINE_COUNT >= LOG_LINES_PER_FILE
                or not CURRENT_LOG_FILE.exists()
            )
            if needs_new_file:
                stamp = dt.datetime.now(US_EASTERN).strftime("%Y%m%d-%H%M%S")
                suffix = uuid.uuid4().hex[:6]
                CURRENT_LOG_FILE = LOG_ARCHIVE_DIR / f"log-{PROCESS_ID}-{stamp}-{suffix}.txt"
                CURRENT_LOG_LINE_COUNT = 0
            with CURRENT_LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
            CURRENT_LOG_LINE_COUNT += 1
            _enforce_log_rotation()
    except OSError:
        return


def load_recent_logs(max_entries: int = 500) -> list[str]:
    """Load the most recent log entries from archived log files."""

    try:
        files = sorted(
            LOG_ARCHIVE_DIR.glob(f"log-{PROCESS_ID}-*.txt"),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return []

    buffer: deque[str] = deque(maxlen=max_entries)
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    cleaned = line.rstrip("\r\n")
                    if cleaned:
                        buffer.append(cleaned)
        except OSError:
            continue

    return list(buffer)


def _has_valid_ft_session(session_state: dict[str, T.Any] | None) -> bool:
    if not isinstance(session_state, dict):
        return False
    sid = str(session_state.get("sid") or "").strip()
    ftat = str(session_state.get("ftat") or "").strip()
    return bool(sid and ftat)


def _earnings_archive_dir(date_value: dt.date) -> Path:
    return EARNINGS_ARCHIVE_DIR / date_value.isoformat()


def _earnings_archive_file(date_value: dt.date) -> Path:
    return _earnings_archive_dir(date_value) / "summary.json"


def _earnings_symbol_file(date_value: dt.date, symbol: str) -> Path:
    safe_symbol = "".join(
        ch for ch in str(symbol) if ch.isalnum() or ch in {"-", "_", "."}
    )
    safe_symbol = safe_symbol or "symbol"
    return _earnings_archive_dir(date_value) / f"{safe_symbol}.json"


def _prediction_archive_dir(date_value: dt.date) -> Path:
    return PREDICTION_ARCHIVE_DIR / date_value.isoformat()


def _prediction_archive_file(date_value: dt.date) -> Path:
    return _prediction_archive_dir(date_value) / "summary.json"


def _prediction_symbol_dir(date_value: dt.date) -> Path:
    return _prediction_archive_dir(date_value)


def _prediction_symbol_file(date_value: dt.date, symbol: str) -> Path:
    safe_symbol = "".join(
        ch for ch in symbol if ch.isalnum() or ch in {"-", "_", "."}
    )
    safe_symbol = safe_symbol or "symbol"
    return _prediction_symbol_dir(date_value) / f"{safe_symbol}.json"


def _describe_timeline_task(target_date: dt.date, timeline_cfg: dict[str, T.Any]) -> tuple[str, str, str]:
    key = str(timeline_cfg.get("key"))
    lookback = int(timeline_cfg.get("lookback", 0) or 0)
    offset_label = f"T{lookback:+d}"
    name = f"{target_date.strftime('%m月%d日')} 的 {offset_label} 天预测"
    return f"predict::{key}", name, offset_label


def _resolve_timeline_offset_days(timeline_cfg: dict[str, T.Any]) -> int:
    """Return the calendar-day offset configured for a prediction timeline."""

    offset = timeline_cfg.get("offset_days")
    if isinstance(offset, (int, float)):
        try:
            return int(offset)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return int(float(offset)) if offset is not None else 0
    lookback = timeline_cfg.get("lookback")
    if isinstance(lookback, (int, float)):
        try:
            return int(lookback)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return int(float(lookback)) if lookback is not None else 0
    return 0

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

PREDICTION_TASK_SEQUENCE = [
    "decision_day",
    "plus1",
    "plus3",
    "plus7",
    "plus14",
]

TASK_ORDER = [
    DAILY_T_MINUS_ONE_TASK_ID,
    DAILY_ADJUST_TASK_ID,
    *[f"predict::{key}" for key in PREDICTION_TASK_SEQUENCE],
    BACKFILL_TASK_ID,
]

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
TASK_TEMPLATES: dict[str, dict[str, T.Any]] = {
    DAILY_T_MINUS_ONE_TASK_ID: {
        "id": DAILY_T_MINUS_ONE_TASK_ID,
        "name": "T-1 实际结果检验",
    },
    DAILY_ADJUST_TASK_ID: {
        "id": DAILY_ADJUST_TASK_ID,
        "name": "模型参数及因子权重调整",
    },
    BACKFILL_TASK_ID: {
        "id": BACKFILL_TASK_ID,
        "name": "回溯验证任务",
    },
}

for _timeline_cfg in PREDICTION_TIMELINES:
    _key = str(_timeline_cfg.get("key"))
    if not _key:
        continue
    _task_id = f"predict::{_key}"
    _default_name = f"{_timeline_cfg.get('label', _key)} 预测"
    if _key == "decision_day":
        _default_name = "T+0 预测"
    elif _key == "plus1":
        _default_name = "T+1 预测"
    elif _key == "plus3":
        _default_name = "T+3 预测"
    elif _key == "plus7":
        _default_name = "T+7 预测"
    elif _key == "plus14":
        _default_name = "T+14 预测"
    TASK_TEMPLATES[_task_id] = {
        "id": _task_id,
        "name": _default_name,
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
        if entry.get("options_filter_applied") is not True:
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
    if options_filter_applied is not True:
        return
    key = date_value.isoformat()
    generated_at = dt.datetime.now(US_EASTERN).isoformat()
    payload = {
        "rowData": row_data,
        "status": status,
        "generated_at": generated_at,
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

        symbol_dir = _earnings_archive_dir(date_value)
        symbol_map: dict[str, list[dict[str, T.Any]]] = {}
        for entry in row_data or []:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            symbol_map.setdefault(symbol, []).append(dict(entry))

        if symbol_map:
            symbol_dir.mkdir(parents=True, exist_ok=True)
            for symbol, rows in symbol_map.items():
                file_path = _earnings_symbol_file(date_value, symbol)
                try:
                    with file_path.open("r", encoding="utf-8") as fh:
                        existing_payload = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    existing_payload = {}

                history = existing_payload.get("history")
                if not isinstance(history, list):
                    history = []

                if rows:
                    rows_payload = rows
                else:
                    existing_rows = existing_payload.get("rows")
                    rows_payload = existing_rows if isinstance(existing_rows, list) else []
                rows_safe = _json_safe(rows_payload)

                history_entry = {
                    "generated_at": generated_at,
                    "status": status,
                    "rows": rows_safe,
                }
                if options_filter_applied is not None:
                    history_entry["options_filter_applied"] = bool(options_filter_applied)

                history.append(history_entry)

                symbol_payload = {
                    "decision_date": key,
                    "symbol": symbol,
                    "rows": rows_safe,
                    "latest_generated_at": generated_at,
                    "latest_status": status,
                    "history": _json_safe(history),
                    "total_runs": len(history),
                }
                if options_filter_applied is not None:
                    symbol_payload["options_filter_applied"] = bool(options_filter_applied)

                _write_archive_file(file_path, symbol_payload)

        legacy_path = EARNINGS_ARCHIVE_DIR / f"{key}.json"
        if legacy_path.exists():
            try:
                legacy_path.unlink()
            except OSError:
                pass

        cache = _load_earnings_cache_raw()
        cache[key] = payload_to_write
        try:
            EARNINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    timeline_sources: dict[str, str] | None = None,
    run_identifier: str | None = None,
    *,
    missing: list[str] | None = None,
    errors: list[str] | None = None,
    timeline_statuses: dict[str, dict[str, T.Any]] | None = None,
) -> None:
    key = target_date.isoformat()
    generated_at = dt.datetime.now(US_EASTERN).isoformat()
    run_id = run_identifier or uuid.uuid4().hex

    sorted_results = sorted(
        [entry for entry in results or [] if isinstance(entry, dict)],
        key=lambda item: (
            item.get("symbol", ""),
            item.get("timeline_key", ""),
            item.get("lookback_days", 0),
        ),
    )
    payload = {
        "rowData": _json_safe(row_data),
        "results": _json_safe(sorted_results),
        "status": status,
        "generated_at": generated_at,
        "run_id": run_id,
    }
    if timeline_sources:
        payload["timeline_sources"] = timeline_sources

    missing_entries: list[str] = []
    missing_symbol_set: set[str] = set()
    for item in missing or []:
        if item is None:
            continue
        text = str(item)
        if text not in missing_entries:
            missing_entries.append(text)
        base = text.split("(", 1)[0].strip().upper()
        if base:
            missing_symbol_set.add(base)

    error_entries: list[str] = []
    error_symbol_set: set[str] = set()
    for item in errors or []:
        if item is None:
            continue
        text = str(item)
        if text not in error_entries:
            error_entries.append(text)
        base = text.split("(", 1)[0].strip().upper()
        if base:
            error_symbol_set.add(base)

    payload["missing"] = _json_safe(missing_entries)
    payload["errors"] = _json_safe(error_entries)
    if timeline_statuses:
        payload["timeline_statuses"] = _json_safe(timeline_statuses)
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

        symbol_dir = _prediction_symbol_dir(target_date)

        symbol_groups: dict[str, list[dict[str, T.Any]]] = {}
        for entry in sorted_results:
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            symbol_groups.setdefault(symbol, []).append(entry)

        metadata_lookup: dict[str, dict[str, T.Any]] = {}
        for meta in row_data or []:
            if not isinstance(meta, dict):
                continue
            symbol = str(meta.get("symbol") or "").upper()
            if symbol:
                metadata_lookup[symbol] = meta

        all_symbols = (
            set(symbol_groups.keys())
            | set(metadata_lookup.keys())
            | missing_symbol_set
            | error_symbol_set
        )
        if all_symbols:
            symbol_dir.mkdir(parents=True, exist_ok=True)
            for symbol in sorted(all_symbols):
                symbol_key = str(symbol).upper()
                entries = symbol_groups.get(symbol_key, [])
                file_path = _prediction_symbol_file(target_date, symbol_key)
                legacy_path = symbol_dir / f"{key}_{symbol_key}.json"

                existing_payload: dict[str, T.Any] = {}
                legacy_source_used = False
                for candidate in (file_path, legacy_path):
                    try:
                        with candidate.open("r", encoding="utf-8") as fh:
                            payload_candidate = json.load(fh)
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(payload_candidate, dict):
                        existing_payload = payload_candidate
                        legacy_source_used = candidate == legacy_path
                        break

                history = existing_payload.get("history")
                if not isinstance(history, list):
                    history = []

                entry_results_safe = _json_safe(entries)
                latest_timelines = {}
                for entry in entries:
                    timeline_key = str(entry.get("timeline_key") or "")
                    if timeline_key:
                        latest_timelines[timeline_key] = entry

                symbol_missing = symbol_key in missing_symbol_set
                symbol_error = symbol_key in error_symbol_set
                status_flag = "missing" if symbol_missing else "error" if symbol_error else "completed"

                history_entry = {
                    "run_id": run_id,
                    "generated_at": generated_at,
                    "status": status_flag,
                    "results": entry_results_safe,
                    "timelines": _json_safe(latest_timelines),
                    "missing": symbol_missing,
                    "error": symbol_error,
                }
                if timeline_sources:
                    history_entry["timeline_sources"] = timeline_sources

                history.append(history_entry)

                metadata = metadata_lookup.get(symbol_key)
                if metadata is None and isinstance(existing_payload.get("metadata"), dict):
                    metadata = existing_payload["metadata"]
                metadata_safe = _json_safe(metadata) if metadata is not None else None

                symbol_rows = [
                    row
                    for row in row_data or []
                    if isinstance(row, dict)
                    and str(row.get("symbol") or "").upper() == symbol_key
                ]
                if symbol_rows:
                    rows_payload = symbol_rows
                else:
                    existing_rows = existing_payload.get("rows")
                    rows_payload = existing_rows if isinstance(existing_rows, list) else []
                rows_safe = _json_safe(rows_payload)

                per_symbol_payload = {
                    "decision_date": key,
                    "symbol": symbol_key,
                    "history": _json_safe(history),
                    "total_runs": len(history),
                    "latest_run_id": run_id,
                    "latest_generated_at": generated_at,
                    "latest_status": status_flag,
                    "latest_results": entry_results_safe,
                    "latest_timelines": _json_safe(latest_timelines),
                    "rows": rows_safe,
                    "missing": symbol_missing,
                    "error": symbol_error,
                }
                if metadata_safe:
                    per_symbol_payload["metadata"] = metadata_safe
                if timeline_sources:
                    per_symbol_payload["timeline_sources"] = timeline_sources

                for preserved_key in ("evaluation", "actuals"):
                    if preserved_key in existing_payload:
                        per_symbol_payload[preserved_key] = existing_payload[preserved_key]

                _write_archive_file(file_path, per_symbol_payload)

                if legacy_source_used and legacy_path.exists() and legacy_path != file_path:
                    try:
                        legacy_path.unlink()
                    except OSError:
                        pass

        legacy_summary = PREDICTION_ARCHIVE_DIR / f"{key}.json"
        if legacy_summary.exists():
            try:
                legacy_summary.unlink()
            except OSError:
                pass

        archive = _load_prediction_archive_raw()
        archive[key] = payload_to_write
        try:
            with PREDICTION_ARCHIVE_PATH.open("w", encoding="utf-8") as fh:
                json.dump(archive, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass


def _group_rowdata_by_timeline(
    row_data: T.Any,
) -> dict[str, dict[str, dict[str, T.Any]]]:
    grouped: dict[str, dict[str, dict[str, T.Any]]] = {}
    if not isinstance(row_data, list):
        return grouped
    for entry in row_data:
        if not isinstance(entry, dict):
            continue
        timeline_key = str(entry.get("timeline_key") or "")
        symbol = str(entry.get("symbol") or "").upper()
        if not timeline_key or not symbol:
            continue
        grouped.setdefault(timeline_key, {})[symbol] = dict(entry)
    return grouped


def _group_results_by_timeline(
    results: T.Any,
) -> dict[str, list[dict[str, T.Any]]]:
    grouped: dict[str, list[dict[str, T.Any]]] = {}
    if not isinstance(results, list):
        return grouped
    for entry in results:
        if not isinstance(entry, dict):
            continue
        timeline_key = str(entry.get("timeline_key") or "")
        if not timeline_key:
            continue
        grouped.setdefault(timeline_key, []).append(dict(entry))
    return grouped


def _extract_timeline_statuses(
    archive_entry: dict[str, T.Any] | None,
) -> dict[str, dict[str, T.Any]]:
    statuses: dict[str, dict[str, T.Any]] = {}
    if not isinstance(archive_entry, dict):
        return statuses

    raw_statuses = archive_entry.get("timeline_statuses")
    if isinstance(raw_statuses, dict):
        for key, payload in raw_statuses.items():
            if not isinstance(payload, dict):
                continue
            state = str(payload.get("state") or "").lower()
            if not state:
                continue
            detail = str(payload.get("detail") or "")
            symbol_count = payload.get("symbol_count")
            try:
                symbol_count_int = int(symbol_count) if symbol_count is not None else 0
            except (TypeError, ValueError):
                symbol_count_int = 0
            statuses[str(key)] = {
                "state": state,
                "detail": detail,
                "symbol_count": symbol_count_int,
                "source_date": str(payload.get("source_date") or ""),
                "updated_at": str(payload.get("updated_at") or ""),
            }

    timeline_sources = archive_entry.get("timeline_sources")
    row_groups = _group_rowdata_by_timeline(archive_entry.get("rowData"))
    result_groups = _group_results_by_timeline(archive_entry.get("results"))

    for cfg in PREDICTION_TIMELINES:
        key = str(cfg.get("key") or "")
        if not key:
            continue
        entry = statuses.get(key)
        source_date = ""
        if isinstance(timeline_sources, dict):
            source_date = str(timeline_sources.get(key) or "")

        symbol_count = len(row_groups.get(key, {}))
        result_count = len(result_groups.get(key, []))

        if entry is None and result_count > 0:
            statuses[key] = {
                "state": "completed",
                "detail": f"存档中已有 {result_count} 条预测结果",
                "symbol_count": symbol_count or result_count,
                "source_date": source_date,
                "updated_at": "",
            }
            entry = statuses[key]

        if entry is None and source_date:
            statuses[key] = {
                "state": "empty",
                "detail": "存档标记为已运行但无有效预测",
                "symbol_count": symbol_count,
                "source_date": source_date,
                "updated_at": "",
            }
            entry = statuses[key]

        if entry is not None and not entry.get("detail"):
            if entry.get("state") == "completed":
                entry["detail"] = "存档中已有预测结果"
            elif entry.get("state") == "empty":
                entry["detail"] = "存档中记录为空结果"

    return statuses


def _timeline_state_is_done(status: dict[str, T.Any] | None) -> bool:
    if not isinstance(status, dict):
        return False
    state = str(status.get("state") or "").lower()
    return state in {"completed"}


def _build_task_updates_from_statuses(
    target_date: dt.date,
    status_map: dict[str, dict[str, T.Any]],
) -> list[dict[str, T.Any]]:
    updates: list[dict[str, T.Any]] = []
    for cfg in PREDICTION_TIMELINES:
        timeline_key = str(cfg.get("key") or "")
        if not timeline_key:
            continue
        status_info = status_map.get(timeline_key)
        if not status_info:
            continue
        state = str(status_info.get("state") or "").lower()
        if state not in {"completed", "empty", "failed"}:
            continue
        offset_days = _resolve_timeline_offset_days(cfg)
        timeline_date = target_date + dt.timedelta(days=offset_days)
        offset_label = str(cfg.get("label") or cfg.get("key") or "未知时点")
        detail_body = status_info.get("detail") or (
            "存档中已有预测结果" if state == "completed" else "存档中无可用预测"
        )
        detail_text = f"{offset_label}｜决策日 {timeline_date.isoformat()}：{detail_body}"
        task_id, name, _ = _describe_timeline_task(target_date, cfg)
        symbol_count = int(status_info.get("symbol_count") or 0)
        if state == "completed":
            status_label = "已完成"
            completed_symbols = symbol_count
        elif state == "empty":
            status_label = "无数据"
            completed_symbols = 0
        else:
            status_label = "失败"
            completed_symbols = 0
        updates.append(
            {
                "id": task_id,
                "name": name,
                "status": status_label,
                "detail": detail_text,
                "total_symbols": symbol_count,
                "completed_symbols": completed_symbols,
                "processed_symbols": completed_symbols,
                "end_time": status_info.get("updated_at") or "",
                "start_time": "",
            }
        )
    for entry in updates:
        entry.setdefault("detail", "")
        entry.setdefault("total_symbols", 0)
        entry.setdefault("completed_symbols", 0)
        entry.setdefault("processed_symbols", 0)
    return updates


def _build_prediction_store_from_archive(
    archive_entry: dict[str, T.Any] | None,
    snapshot: T.Any,
) -> dict[str, T.Any] | None:
    if not isinstance(archive_entry, dict):
        return None

    results: list[dict[str, T.Any]] = []
    raw_results = archive_entry.get("results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                results.append(item)

    missing_set: set[str] = set()
    raw_missing = archive_entry.get("missing")
    if isinstance(raw_missing, list):
        for entry in raw_missing:
            if entry is None:
                continue
            missing_set.add(str(entry))

    errors_set: set[str] = set()
    raw_errors = archive_entry.get("errors")
    if isinstance(raw_errors, list):
        for entry in raw_errors:
            if entry is None:
                continue
            errors_set.add(str(entry))

    return {
        "results": results,
        "missing": sorted(missing_set),
        "errors": sorted(errors_set),
        "rl_snapshot": snapshot,
    }


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

        legacy_summary = PREDICTION_ARCHIVE_DIR / f"{key}.json"
        if legacy_summary.exists():
            try:
                legacy_summary.unlink()
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
        SYMBOL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SYMBOL_META_PATH.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _fetch_symbol_profile(symbol: str) -> dict[str, T.Any] | None:
    return finnhub.fetch_company_profile(symbol)


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
DEFAULT_PICKER_DATE, MIN_PICKER_DATE, MAX_PICKER_DATE = _date_picker_bounds()




def _load_dci_payloads() -> dict[str, dict[str, T.Any]]:
    payload = load_dci_payloads()
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


def _build_baseline_dci_snapshot() -> dict[str, T.Any]:
    """返回一个使用内置中性参数的 DCI 输入快照。"""

    factors = {name: {"z": 0.0} for name in get_factor_weights().keys()}
    return {
        "factors": factors,
        "z_cons": 0.0,
        "z_narr": 0.0,
        "CI": 0.0,
        "Q": 0.0,
        "D": 0.0,
        "EM_pct": 5.0,
        "S_stab": 1.0,
        "shock_flag": 0,
        "__source__": "baseline",
    }


def _build_baseline_symbol_payload(
    timelines: list[dict[str, T.Any]] | None,
) -> dict[str, T.Any]:
    """构造包含所有时点默认快照的标的输入。"""

    base_snapshot = _build_baseline_dci_snapshot()
    payload: dict[str, T.Any] = dict(base_snapshot)
    snapshots: dict[str, dict[str, T.Any]] = {}
    if timelines:
        for cfg in timelines:
            key = str(cfg.get("key") or "").strip()
            if not key:
                continue
            snapshots[key] = _build_baseline_dci_snapshot()
    payload["snapshots"] = snapshots
    payload["__source__"] = "baseline"
    return payload


def _summarise_dci_payloads(
    symbols: list[str],
    payloads: dict[str, dict[str, T.Any]] | None,
) -> tuple[list[str], list[str], list[str]]:
    """根据已载入的 DCI 输入区分可用、缺失及回退的标的。"""

    available: list[str] = []
    missing: list[str] = []
    fallback: list[str] = []

    if not isinstance(payloads, dict):
        payloads = {}

    for raw_symbol in symbols:
        symbol = (raw_symbol or "").upper()
        if not symbol:
            continue
        if isinstance(payloads.get(symbol), dict):
            available.append(symbol)
        elif DCI_AUTO_BASELINE_ENABLED:
            fallback.append(symbol)
        else:
            missing.append(symbol)

    return available, missing, fallback


def _compute_dci_for_symbols(
    symbols: list[str],
    metadata_map: dict[str, dict[str, T.Any]] | None = None,
    progress_callback: T.Optional[
        T.Callable[[str, dict[str, T.Any], str, T.Optional[str]], None]
    ] = None,
    timeline_configs: list[dict[str, T.Any]] | None = None,
    payloads: dict[str, dict[str, T.Any]] | None = None,
) -> tuple[
    list[dict[str, T.Any]],
    list[str],
    list[str],
    dict[str, str],
]:
    if not isinstance(payloads, dict):
        payloads = _load_dci_payloads()
    results: list[dict[str, T.Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    missing_detail_map: dict[str, str] = {}

    timelines = timeline_configs or PREDICTION_TIMELINES

    for raw_symbol in symbols:
        symbol = (raw_symbol or "").upper()
        if not symbol:
            continue
        data = payloads.get(symbol)
        symbol_uses_baseline = False
        if not data and DCI_AUTO_BASELINE_ENABLED:
            data = _build_baseline_symbol_payload(timelines)
            symbol_uses_baseline = True
        if not data:
            if progress_callback:
                for timeline in timelines:
                    try:
                        progress_callback(symbol, timeline, "missing", None)
                    except Exception:  # pragma: no cover - logging best effort
                        pass
            missing.append(symbol)
            continue

        meta_entry = metadata_map.get(symbol) if isinstance(metadata_map, dict) else None
        decision_date = str((meta_entry or {}).get("decision_date") or "")
        timeline_date_value = str((meta_entry or {}).get("timeline_date") or decision_date)
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
            timeline_uses_baseline = symbol_uses_baseline
            if not payload and DCI_AUTO_BASELINE_ENABLED:
                payload = _build_baseline_dci_snapshot()
                alias = str(timeline.get("key") or alias)
                timeline_uses_baseline = True
            if not payload:
                label = str(timeline.get("label") or timeline.get("key") or "未知时点")
                aliases = [str(timeline.get("key") or "")] + [
                    str(item)
                    for item in (timeline.get("aliases") or [])
                    if item
                ]
                alias_part = "，".join([item for item in aliases if item]) or "无"
                available_keys: list[str] = []
                snapshots = data.get("snapshots")
                if isinstance(snapshots, dict):
                    for snap_key, snap_val in snapshots.items():
                        if isinstance(snap_key, str) and isinstance(snap_val, dict):
                            available_keys.append(snap_key)
                for direct_key, direct_val in data.items():
                    if isinstance(direct_key, str) and isinstance(direct_val, dict):
                        available_keys.append(direct_key)
                available_keys = sorted({key for key in available_keys if key})
                sample_available = "，".join(available_keys[:6])
                if len(available_keys) > 6:
                    sample_available += "……"
                available_part = sample_available or "无"
                reason = (
                    f"缺少 {label} 的输入快照（尝试：{alias_part}；已有：{available_part}）"
                )
                if progress_callback:
                    try:
                        progress_callback(symbol, timeline, "missing", reason)
                    except Exception:  # pragma: no cover - logging best effort
                        pass
                identifier = f"{symbol}({label})"
                missing.append(identifier)
                missing_detail_map[identifier] = reason
                continue

            try:
                inputs = build_inputs(symbol, payload)
                dci_result = compute_dci(inputs)

                should_trade = dci_result.dci_final >= 60.0
                base_direction = "放弃"
                if should_trade:
                    base_direction = "多" if dci_result.direction > 0 else "空"

                rl_direction = "放弃" if not should_trade else ""
                rl_p_up_pct: float | None = None
                rl_delta_pct: float | None = None
                rl_prediction_id = ""
                selection_value = f"{symbol}::{timeline.get('key')}"

                if should_trade and RL_MANAGER is not None:
                    try:
                        rl_pred = RL_MANAGER.record_prediction(
                            dci_result,
                            sector=sector_key,
                        )
                        rl_direction = "多" if rl_pred.direction > 0 else "空"
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
                        "timeline_offset_days": int(timeline.get('lookback', 0)),
                        "timeline_date": timeline_date_value,
                        "decision_date": decision_date,
                        "bucket": bucket,
                        "company": company_name,
                        "sector": sector_value or "未知",
                        "direction": base_direction,
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
                        "input_source": "baseline"
                        if timeline_uses_baseline
                        else "dataset",
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
    return results, missing, errors, missing_detail_map


def _check_resource_connections(ft_session: dict[str, T.Any] | None) -> list[dict[str, T.Any]]:
    statuses: list[dict[str, T.Any]] = []

    def add_status(name: str, parameter: str, ok: bool, detail: str) -> None:
        statuses.append(
            {
                "resource": name,
                "parameter": parameter,
                "ok": ok,
                "detail": detail,
            }
        )

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
        return nasdaq.check_status(today_str)

    ok, detail = run_with_retry(_check_nasdaq_once)
    add_status("Nasdaq 财报 API", "财报列表（代码/时间段）", ok, detail)

    # Finnhub API
    def _check_finnhub_once() -> tuple[bool, str]:
        return finnhub.check_status("AAPL")

    ok, detail = run_with_retry(_check_finnhub_once)
    add_status("Finnhub API", "实时行情（现价/昨收）", ok, detail)

    # OpenFIGI API
    def _check_openfigi_once() -> tuple[bool, str]:
        return openfigi.check_status("AAPL")

    ok, detail = run_with_retry(_check_openfigi_once)
    add_status("OpenFIGI API", "Ticker → FIGI 映射", ok, detail)

    # FRED API
    def _check_fred_once() -> tuple[bool, str]:
        return fred.check_status("DGS3MO")

    ok, detail = run_with_retry(_check_fred_once)
    add_status("FRED API", "DGS3MO 系列（最新观测值）", ok, detail)

    # Firstrade session status
    if isinstance(ft_session, dict) and ft_session:
        sid = str(ft_session.get("sid", ""))
        ok = bool(ft_session.get("sid"))
        detail = f"已缓存会话 (sid {sid[:4]}...)" if ok else "会话信息不完整"
    else:
        ok = False
        detail = "尚未登录或无会话信息"
    add_status("Firstrade 会话", "周五期权筛选凭证", ok, detail)

    return statuses


def _format_decimal(value: T.Any, digits: int = 4) -> str:
    """Format numeric values with trimmed trailing zeros."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—" if value in {None, ""} else str(value)

    if math.isnan(number) or math.isinf(number):
        return str(number)

    formatted = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _extract_factor_source(
    factor_payload: T.Any, symbol_payload: dict[str, T.Any]
) -> str:
    """Best-effort extraction of the data source tag for a factor."""

    if isinstance(factor_payload, dict):
        for key in ("source", "__source__", "provider"):
            candidate = factor_payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    for key in ("source", "__source__", "provider"):
        candidate = symbol_payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return ""


def _format_factor_value(payload: T.Any) -> str:
    """Render the main numeric content for a factor payload."""

    if isinstance(payload, dict):
        parts: list[str] = []

        if payload.get("z") is not None:
            parts.append(f"z={_format_decimal(payload.get('z'))}")

        value_present = payload.get("value") is not None
        if value_present:
            parts.append(f"value={_format_decimal(payload.get('value'))}")

        median = payload.get("median")
        mad = payload.get("mad")
        extras: list[str] = []
        if value_present and median is not None:
            extras.append(f"median={_format_decimal(median)}")
        if value_present and mad is not None:
            extras.append(f"mad={_format_decimal(mad)}")
        if extras:
            parts.append(" / ".join(extras))

        if not parts:
            for key, candidate in payload.items():
                if key in {"z", "value", "median", "mad"}:
                    continue
                if isinstance(candidate, (int, float)):
                    parts.append(f"{key}={_format_decimal(candidate)}")
            if not parts and payload:
                parts.append(str(payload))

        return "；".join(parts) if parts else "—"

    if isinstance(payload, (int, float)):
        return _format_decimal(payload)

    if payload in {None, ""}:
        return "—"

    return str(payload)


def _collect_factor_source_rows() -> tuple[list[dict[str, str]], str | None]:
    """Return rows describing factor values and their origins."""

    payloads = _load_dci_payloads()
    factor_names = list(BASE_FACTOR_WEIGHTS.keys())

    if not payloads:
        rows = [
            {"factor": name, "value": "—", "source": "未载入数据"}
            for name in factor_names
        ]
        return rows, "未载入任何 DCI 输入数据，显示占位结果。"

    sorted_payloads = sorted(payloads.items(), key=lambda item: str(item[0]))
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for factor_name in factor_names:
        best_with_source: dict[str, str] | None = None
        fallback_entry: dict[str, str] | None = None

        for symbol, symbol_payload in sorted_payloads:
            if not isinstance(symbol_payload, dict):
                continue
            factors = symbol_payload.get("factors")
            if not isinstance(factors, dict):
                continue
            factor_payload = factors.get(factor_name)
            if factor_payload is None:
                continue

            source = _extract_factor_source(factor_payload, symbol_payload)
            value_text = _format_factor_value(factor_payload)
            if symbol:
                value_text = f"{value_text}｜{symbol}"

            entry = {
                "factor": factor_name,
                "value": value_text,
                "source": source or "—",
            }

            if source:
                best_with_source = entry
                break
            if fallback_entry is None:
                fallback_entry = entry

        if best_with_source:
            rows.append(best_with_source)
        elif fallback_entry:
            rows.append(fallback_entry)
        else:
            rows.append({"factor": factor_name, "value": "—", "source": "—"})
            missing.append(factor_name)

    note: str | None = None
    if missing:
        missing_text = "，".join(missing)
        note = f"以下因子未找到数据：{missing_text}"

    return rows, note


def _render_factor_preview_table(rows: list[dict[str, str]]) -> html.Div:
    """Split rows into three columns of tables for compact display."""

    columns: list[list[dict[str, str]]] = [[], [], []]
    for idx, row in enumerate(rows):
        columns[idx % 3].append(row)

    column_components: list[dbc.Col] = []
    for col_rows in columns:
        if not col_rows:
            continue
        table_rows = [
            html.Tr(
                [
                    html.Td(entry.get("factor", "")),
                    html.Td(entry.get("value", "")),
                    html.Td(entry.get("source", "")),
                ]
            )
            for entry in col_rows
        ]
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("因子"),
                        html.Th("数值"),
                        html.Th("数据源"),
                    ])
                ),
                html.Tbody(table_rows),
            ],
            bordered=True,
            hover=True,
            size="sm",
            className="mb-3",
        )
        column_components.append(dbc.Col(table, width=12, lg=4))

    return dbc.Row(column_components, className="g-2")


def _resolve_payload_source(payload: dict[str, T.Any]) -> str:
    for key in ("source", "__source__", "provider"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _format_preview_value(value: T.Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return _format_decimal(value)
    if value in {None, ""}:
        return "—"
    return str(value)


def _build_preview_rows_for_symbol(
    symbol: str,
    payload: dict[str, T.Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source = _resolve_payload_source(payload) or "—"

    meta_fields = [
        ("symbol", "标的代码"),
        ("company", "公司"),
        ("sector", "行业"),
        ("industry", "子行业"),
        ("exchange", "交易所"),
        ("bucket", "时间段"),
        ("decision_date", "决策日"),
    ]
    for key, label in meta_fields:
        if key == "symbol":
            rows.append({"factor": label, "value": symbol, "source": source})
            continue
        value = payload.get(key)
        if value in {None, ""}:
            continue
        rows.append({"factor": label, "value": _format_preview_value(value), "source": source})

    metric_fields = [
        ("z_cons", "一致性 z"),
        ("z_narr", "叙事 z"),
        ("CI", "拥挤度"),
        ("Q", "质量分"),
        ("D", "分歧度"),
        ("EM_pct", "预期波动(%)"),
        ("S_stab", "稳定性"),
        ("shock_flag", "宏观冲击标记"),
        ("has_weekly", "有周度期权"),
        ("option_expiry_days", "期权到期日(天)"),
        ("option_expiry_type", "到期类型"),
    ]

    for key, label in metric_fields:
        value = payload.get(key)
        rows.append({"factor": label, "value": _format_preview_value(value), "source": source})

    factors = payload.get("factors")
    if isinstance(factors, dict):
        for factor in BASE_FACTOR_WEIGHTS:
            factor_payload = factors.get(factor)
            value_text = _format_factor_value(factor_payload)
            factor_source = _extract_factor_source(factor_payload, payload) or source
            rows.append(
                {
                    "factor": factor,
                    "value": value_text,
                    "source": factor_source,
                }
            )

    return rows


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
                    html.Td(entry.get("parameter", "")),
                    html.Td(badge),
                    html.Td(entry.get("detail", "")),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("资源"),
                    html.Th("参数"),
                    html.Th("状态"),
                    html.Th("详情"),
                ])
            ),
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

    factor_rows, factor_note = _collect_factor_source_rows()
    if factor_rows:
        content.append(html.Hr())
        content.append(
            html.H5("因子数据提取预览", style={"marginTop": "12px", "fontWeight": "bold"})
        )
        if factor_note:
            content.append(
                html.Div(
                    factor_note,
                    className="text-muted",
                    style={"marginBottom": "8px"},
                )
            )
        content.append(_render_factor_preview_table(factor_rows))
    elif factor_note:
        content.append(html.Hr())
        content.append(
            html.Div(
                factor_note,
                className="text-muted",
                style={"marginTop": "12px"},
            )
        )
    return html.Div(content)


def build_prediction_preview(
    task_state: dict[str, T.Any] | None = None,
    prediction_store: dict[str, T.Any] | None = None,
    agent_data: dict[str, T.Any] | None = None,
    *,
    force_reload: bool = False,
) -> tuple[str, T.Any, T.Any]:
    """Summarise the latest prediction task status and key parameters."""

    del force_reload  # Manual refresh is handled by Dash trigger semantics.

    tasks: list[dict[str, T.Any]] = []
    target_date = ""
    if isinstance(task_state, dict):
        target_date = str(task_state.get("target_date") or "")
        raw_tasks = task_state.get("tasks")
        if isinstance(raw_tasks, list):
            for entry in raw_tasks:
                if isinstance(entry, dict):
                    tasks.append(dict(entry))

    order_map = {task_id: idx for idx, task_id in enumerate(TASK_ORDER)}
    tasks.sort(
        key=lambda item: (
            order_map.get(str(item.get("id") or ""), len(order_map)),
            str(item.get("start_time") or "9999"),
            str(item.get("name") or ""),
        )
    )

    status_counter: Counter[str] = Counter()
    last_update = ""
    for entry in tasks:
        status_value = str(entry.get("status") or "等待")
        status_counter[status_value] += 1
        updated_at = str(entry.get("updated_at") or "")
        if updated_at and (not last_update or updated_at > last_update):
            last_update = updated_at

    results: list[dict[str, T.Any]] = []
    missing_entries: list[str] = []
    error_entries: list[str] = []
    if isinstance(prediction_store, dict):
        raw_results = prediction_store.get("results")
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    results.append(item)
        raw_missing = prediction_store.get("missing")
        if isinstance(raw_missing, list):
            missing_entries = [str(entry) for entry in raw_missing if entry is not None]
        raw_errors = prediction_store.get("errors")
        if isinstance(raw_errors, list):
            error_entries = [str(entry) for entry in raw_errors if entry is not None]

    symbol_set = {
        str(entry.get("symbol") or "").upper()
        for entry in results
        if str(entry.get("symbol") or "")
    }
    timeline_set = {
        str(entry.get("timeline_key") or "")
        for entry in results
        if str(entry.get("timeline_key") or "")
    }

    timestamp = dt.datetime.now(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z")

    status_parts: list[str] = []
    if target_date:
        status_parts.append(f"目标预测日：{target_date}")
    if tasks:
        summary_order = ["进行中", "已完成", "等待", "失败", "无数据"]
        summary_text = [
            f"{label}{status_counter[label]}项"
            for label in summary_order
            if status_counter.get(label)
        ]
        if summary_text:
            status_parts.append("步骤概况：" + "，".join(summary_text))
    if last_update:
        status_parts.append(f"最近更新：{last_update}")
    if results:
        status_parts.append(
            f"预测结果：{len(results)} 条｜标的 {len(symbol_set) or '0'} 个｜时点 {len(timeline_set) or '0'} 个"
        )
    if missing_entries:
        status_parts.append(f"缺失数据 {len(missing_entries)} 条")
    if error_entries:
        status_parts.append(f"异常条目 {len(error_entries)} 条")
    if status_parts:
        status_parts.append(f"生成时间：{timestamp}")
        status_message = " ｜ ".join(status_parts)
    else:
        status_message = f"尚未检测到预测任务，请在预测页启动任务。｜生成时间：{timestamp}"

    status_colors = {
        "已完成": "success",
        "进行中": "primary",
        "等待": "secondary",
        "失败": "danger",
        "无数据": "warning",
    }

    if not tasks:
        steps_component = html.Div(
            "当前没有预测任务，请先在预测页启动任务。",
            className="text-muted",
        )
    else:
        step_items: list[html.Li] = []
        for entry in tasks:
            name = str(entry.get("name") or entry.get("id") or "未命名步骤")
            status_value = str(entry.get("status") or "等待")
            detail_text = str(entry.get("detail") or "").strip()
            progress_text = str(entry.get("symbol_progress") or "").strip()
            start_time = str(entry.get("start_time") or "").strip()
            end_time = str(entry.get("end_time") or "").strip()
            updated_at = str(entry.get("updated_at") or "").strip()

            badge = dbc.Badge(
                status_value,
                color=status_colors.get(status_value, "secondary"),
                className="ms-2",
            )

            header = html.Div(
                [
                    html.Span(name, className="fw-semibold"),
                    badge,
                ],
                className="d-flex justify-content-between align-items-center mb-1",
            )

            meta_parts: list[str] = []
            if progress_text and progress_text not in {"", "-"}:
                meta_parts.append(f"进度：{progress_text}")
            if start_time:
                meta_parts.append(f"开始：{start_time}")
            if end_time:
                meta_parts.append(f"结束：{end_time}")
            if updated_at:
                meta_parts.append(f"更新：{updated_at}")

            details: list[T.Any] = []
            if detail_text:
                details.append(html.Div(detail_text, className="small text-muted"))
            if meta_parts:
                details.append(
                    html.Div(" ｜ ".join(meta_parts), className="small text-muted")
                )

            step_items.append(
                html.Li(
                    [header, *details],
                    className="mb-3",
                )
            )

        steps_component = html.Ol(step_items, className="mb-0")

    def _format_param_value(value: T.Any) -> str:
        if isinstance(value, float):
            return _format_decimal(value, 4)
        if isinstance(value, (int,)):
            return str(value)
        if isinstance(value, (list, tuple, set)):
            return "，".join(str(item) for item in value)
        if value in {None, ""}:
            return "—"
        return str(value)

    def _add_param_row(label: str, value: T.Any, description: str) -> None:
        display = _format_param_value(value)
        if display == "—" and value not in {0, 0.0}:
            return
        param_rows.append(
            html.Tr([html.Td(label), html.Td(display), html.Td(description or "")])
        )

    param_rows: list[html.Tr] = []
    if target_date:
        _add_param_row("目标预测日", target_date, "当前预测任务关联的决策日。")
    if tasks:
        _add_param_row("记录的任务数", len(tasks), "已跟踪的预测流程步骤数量。")
        status_desc = {
            "进行中": "当前正在执行的任务数量。",
            "已完成": "已结束并成功完成的任务数量。",
            "等待": "尚未开始的任务数量。",
            "失败": "执行过程中出现异常的任务数量。",
            "无数据": "缺少数据而被跳过的任务数量。",
        }
        for key, desc in status_desc.items():
            if status_counter.get(key):
                _add_param_row(f"{key}任务", status_counter[key], desc)
        if last_update:
            _add_param_row("最近更新时间", last_update, "最后一次记录任务状态的时间。")
    if results:
        _add_param_row("预测结果数量", len(results), "最新预测任务生成的结果条数。")
        if symbol_set:
            _add_param_row("涉及标的数", len(symbol_set), "参与预测的独立标的数量。")
        if timeline_set:
            _add_param_row("覆盖时点数", len(timeline_set), "本次任务覆盖的预测时点数量。")
    if missing_entries:
        _add_param_row("缺失数据条目", len(missing_entries), "因缺少输入数据而未能生成结果的条目数量。")
    if error_entries:
        _add_param_row("异常条目", len(error_entries), "执行过程中出现异常的条目数量。")

    agent_snapshot = agent_data
    if isinstance(agent_data, dict) and "global" in agent_data and isinstance(agent_data["global"], dict):
        agent_snapshot = agent_data.get("global")

    rl_descriptions = RL_PARAM_DESCRIPTIONS
    if isinstance(agent_snapshot, dict):
        for key, label in [
            ("learning_rate", "RL 学习率"),
            ("gamma", "RL 折扣因子"),
            ("adjustment_scale", "概率调整幅度"),
            ("bias", "方向偏置"),
            ("baseline", "RL 基准"),
            ("update_count", "RL 更新次数"),
            ("total_predictions", "RL 预测次数"),
        ]:
            if key in agent_snapshot:
                _add_param_row(label, agent_snapshot.get(key), rl_descriptions.get(key, ""))

        pending_counts = agent_data.get("pending_counts") if isinstance(agent_data, dict) else None
        if not pending_counts and isinstance(agent_snapshot, dict):
            pending_counts = agent_snapshot.get("pending_counts")
        if isinstance(pending_counts, dict) and pending_counts:
            try:
                pending_total = sum(int(value) for value in pending_counts.values())
            except (TypeError, ValueError):
                pending_total = 0
            preview_items = sorted(
                ((str(symbol), int(count)) for symbol, count in pending_counts.items()),
                key=lambda item: (-item[1], item[0]),
            )
            preview_text = "；".join(f"{symbol}:{count}" for symbol, count in preview_items[:5])
            value_text = (
                f"{pending_total}（{preview_text}）" if preview_text and pending_total else str(pending_total)
            )
            _add_param_row("待反馈标的", value_text, "强化学习模块等待实际结果反馈的标的数量。")

        weights = agent_snapshot.get("weights")
        if isinstance(weights, dict) and weights:
            top_weights = sorted(
                ((str(feature), float(weight)) for feature, weight in weights.items()),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            preview = "；".join(
                f"{feature}:{_format_decimal(weight, 4)}" for feature, weight in top_weights[:5]
            )
            _add_param_row("权重预览", preview, rl_descriptions.get("weights", "特征权重。"))

    table_component = (
        dbc.Table(
            [
                html.Thead(html.Tr([html.Th("参数"), html.Th("当前值"), html.Th("说明")])),
                html.Tbody(param_rows),
            ],
            bordered=True,
            hover=True,
            size="sm",
            className="mb-0",
        )
        if param_rows
        else html.Div("暂无可用的预测参数信息。", className="text-muted")
    )

    return status_message, steps_component, table_component


PREDICTION_RUN_LOCK = threading.Lock()
PREDICTION_RUN_STATES: dict[str, dict[str, T.Any]] = {}


def _init_prediction_run_state(
    run_id: str,
    target_date: dt.date,
    logs: list[str] | None,
    task_state: dict[str, T.Any] | None,
    status: str = "",
) -> None:
    payload: dict[str, T.Any] = {
        "logs": list(logs or []),
        "status": status or "",
        "store_data": None,
        "rl_snapshot": None,
        "tasks": copy.deepcopy(task_state) if isinstance(task_state, dict) else None,
        "completed": False,
        "target_date": target_date,
        "error": None,
    }
    with PREDICTION_RUN_LOCK:
        PREDICTION_RUN_STATES[run_id] = payload


def _update_prediction_run_state(run_id: str, **kwargs: T.Any) -> None:
    with PREDICTION_RUN_LOCK:
        state = PREDICTION_RUN_STATES.get(run_id)
        if not state:
            return
        for key, value in kwargs.items():
            if key == "logs":
                state[key] = list(value) if isinstance(value, list) else value
            elif key in {"tasks", "store_data", "rl_snapshot"}:
                if value is None:
                    state[key] = None
                else:
                    try:
                        state[key] = copy.deepcopy(value)
                    except Exception:
                        state[key] = value
            else:
                state[key] = value


def _get_prediction_run_state(run_id: str) -> dict[str, T.Any] | None:
    with PREDICTION_RUN_LOCK:
        state = PREDICTION_RUN_STATES.get(run_id)
        if not state:
            return None
        try:
            return copy.deepcopy(state)
        except Exception:
            return dict(state)


def _clear_prediction_run_state(run_id: str) -> None:
    with PREDICTION_RUN_LOCK:
        PREDICTION_RUN_STATES.pop(run_id, None)

# ---------- Nasdaq API ----------
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


def _shift_weekend_to_monday(base_date: dt.date) -> tuple[dt.date, bool]:
    """Return Monday for weekend dates, indicating whether a shift occurred."""

    wd = base_date.weekday()
    if wd >= 5:  # Saturday/Sunday
        return base_date + dt.timedelta(days=(7 - wd)), True
    return base_date, False


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
    try:
        return nasdaq.fetch_earnings(d)
    except requests.RequestException:
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


def _describe_time_bucket(time_label: str | None) -> str:
    """Map a Nasdaq earnings time string into the dashboard's bucket labels."""

    text = str(time_label or "").strip()
    if _is_after_hours(text):
        return "盘后"
    if _is_pre_market(text):
        return "盘前"
    if _is_time_not_supplied(text):
        return "时间待定"
    return "常规盘"


def append_log(
    logs: T.Optional[T.List[str]], message: str, *, task_label: str | None = None
) -> T.List[str]:
    if not isinstance(logs, list):
        logs = []
    stamp = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
    label = f"[{task_label}] " if task_label else ""
    entry = f"[{stamp}] {label}{message}"
    _persist_log_entry(entry)
    return logs + [entry]
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

    def has_weekly_expiring_on(
        self, symbol: str, expiry: dt.date
    ) -> tuple[T.Optional[bool], T.Optional[str], bool]:
        """Check whether a symbol lists the target Friday contract in Firstrade.

        Returns a tuple ``(has_weekly, expiry_type, matched)`` where:

        - ``has_weekly`` is ``True`` if a weekly option matches the target expiry,
          ``False`` if a non-weekly contract matches, and ``None`` when the lookup
          fails.
        - ``expiry_type`` surfaces the matched ``exp_type`` (such as ``"W"`` or
          ``"M"``) when available.
        - ``matched`` indicates whether Firstrade reported a contract for the
          target expiry regardless of the option type.
        """

        if not self.enabled or not self.session:
            return None, None, False

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
                return None, None, False
            if resp.status_code != 200 or not isinstance(data, dict):
                self._log(f"查询 {symbol.upper()} 的期权失败，HTTP {resp.status_code}。")
                return None, None, False
            err_msg = str(data.get("error") or "").strip()
            if err_msg:
                self._log(f"查询 {symbol.upper()} 的期权返回错误：{err_msg}")
                return None, None, False
            items = data.get("items")
            if not isinstance(items, list):
                return None, None, False

            target = expiry.strftime("%Y%m%d")
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                exp_date = str(entry.get("exp_date", "")).strip()
                if exp_date != target:
                    continue
                exp_type = str(entry.get("exp_type", "")).upper()
                # exp_type "W" => weekly, "M" => monthly, other codes possible
                if exp_type == "W":
                    self._log(
                        f"找到 {symbol.upper()} 在 {target} 的到期日 {exp_date}（类型 {exp_type}）。"
                    )
                    return True, exp_type, True
                self._log(
                    f"找到 {symbol.upper()} 在 {target} 的到期日 {exp_date}（类型 {exp_type}，非周度期权）。"
                )
                return False, exp_type or None, True
            self._log(f"未找到 {symbol.upper()} 在 {target} 的匹配到期日。")
            return False, None, False
        except Exception:
            self._log(f"获取 {symbol.upper()} 期权到期日时出现异常。")
            return None, None, False





def _prepare_earnings_dataset(
    target_date: dt.date,
    session_state: dict[str, T.Any] | None,
    username: str,
    password: str,
    twofa: str,
    *,
    logger: T.Optional[T.Callable[[str], None]] = None,
) -> tuple[list[dict[str, T.Any]], str, dict[str, T.Any] | None, bool, str]:
    """Fetch Nasdaq earnings and keep symbols with contracts on the target Friday."""

    def _log(message: str) -> None:
        if logger:
            try:
                logger(message)
            except Exception:  # pragma: no cover - logging best effort
                pass

    session_payload = session_state if isinstance(session_state, dict) else {}

    try:
        base_rows = fetch_earnings_by_date(target_date)
    except Exception as exc:  # pragma: no cover - defensive network handling
        error = f"财报列表请求失败：{exc}"
        _log(error)
        return [], "", None, False, error

    if not base_rows:
        message = f"{target_date.strftime('%Y-%m-%d')} 未找到可用的财报记录。"
        _log(message)
        return [], message, None, True, ""

    _log(
        f"{target_date.strftime('%Y-%m-%d')} 财报列表共 {len(base_rows)} 条，准备执行 Firstrade 筛选。"
    )

    has_session = _has_valid_ft_session(session_payload)
    ft_client: FTClient | None = None
    session_out: dict[str, T.Any] | None = None

    if has_session or (username and password):
        ft_client = FTClient(
            username,
            password,
            twofa,
            session_state=session_payload if has_session else None,
            login=not has_session,
            logger=_log,
        )
        if ft_client.enabled:
            session_out = ft_client.export_session_state()
        else:
            error = ft_client.error or "Firstrade 会话不可用"
            _log(f"Firstrade 筛选失败：{error}")
            return [], "", session_out, False, error
    else:
        error = "缺少 Firstrade 会话凭证"
        _log(error)
        return [], "", None, False, error

    option_cache: dict[str, tuple[T.Optional[bool], T.Optional[str], bool]] = {}
    option_notes: dict[str, dict[str, T.Any]] = {}
    matched_rows: list[dict[str, T.Any]] = []
    error_message = ""
    options_checked = False
    selection_mode = "pending"

    if ft_client and ft_client.enabled:
        decision_day, _ = _shift_weekend_to_monday(target_date)
        reference_day = next_trading_day(decision_day)
        expiry_date = this_friday(reference_day)
        _log(
            "Firstrade 周五筛选目标：%s（参考日：%s）。"
            % (expiry_date.isoformat(), reference_day.isoformat())
        )
        skipped_symbols: list[str] = []
        missing_symbols: list[str] = []
        non_weekly_symbols: list[str] = []
        weekly_count = 0
        non_weekly_count = 0
        for entry in base_rows:
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            cached = option_cache.get(symbol)
            if cached is None:
                cached = ft_client.has_weekly_expiring_on(symbol, expiry_date)
                option_cache[symbol] = cached
            has_weekly, expiry_type, matched = cached
            if has_weekly is None:
                if symbol not in skipped_symbols:
                    skipped_symbols.append(symbol)
                    _log(f"无法确认 {symbol} 的周五期权信息。")
                continue
            options_checked = True

            notes = option_notes.setdefault(symbol, {})

            if matched:
                notes["weekly"] = bool(has_weekly)
                if expiry_type:
                    notes["expiry_type"] = expiry_type
                matched_rows.append(entry)
                if has_weekly:
                    weekly_count += 1
                else:
                    non_weekly_count += 1
                    label = f"{symbol}（类型 {expiry_type or '未知'}）"
                    if label not in non_weekly_symbols:
                        non_weekly_symbols.append(label)
            else:
                if symbol not in missing_symbols:
                    missing_symbols.append(symbol)

        if skipped_symbols:
            preview = "，".join(skipped_symbols[:5])
            suffix = "…" if len(skipped_symbols) > 5 else ""
            _log(f"部分标的缺少周五期权信息，已跳过：{preview}{suffix}。")

        if matched_rows:
            selection_mode = "matched"
            _log(
                "Firstrade 筛选完成，保留 %d 条记录（周度 %d，其他 %d）。"
                % (len(matched_rows), weekly_count, non_weekly_count)
            )
            base_rows = matched_rows
        else:
            base_rows = []
            if options_checked:
                selection_mode = "empty"
                _log("Firstrade 筛选后暂无符合条件的标的。")
            elif not error_message:
                error_message = "Firstrade 筛选未完成"

        if non_weekly_symbols:
            preview = "，".join(non_weekly_symbols[:5])
            suffix = "…" if len(non_weekly_symbols) > 5 else ""
            _log(f"以下标的非周度期权：{preview}{suffix}。")

        if missing_symbols:
            preview = "，".join(missing_symbols[:5])
            suffix = "…" if len(missing_symbols) > 5 else ""
            _log(f"未找到以下标的的目标到期日：{preview}{suffix}。")
    else:  # pragma: no cover - guarded above
        return [], "", session_out, False, ft_client.error if ft_client else ""

    if error_message and ft_client and not ft_client.enabled and not error_message.endswith("未完成"):
        _log(f"Firstrade 会话失效：{error_message}")
        return [], "", session_out, False, error_message

    if ft_client and ft_client.enabled:
        if selection_mode == "pending":
            status_text = f"{target_date.strftime('%Y-%m-%d')} 未完成 Firstrade 周五期权筛选。"
            return [], status_text, session_out, False, ""
        if selection_mode == "matched":
            status_text = (
                f"{target_date.strftime('%Y-%m-%d')} 筛选周五期权后保留 {len(base_rows)} 个标的。"
            )
        else:
            status_text = f"{target_date.strftime('%Y-%m-%d')} 筛选后暂无符合条件的标的。"
    else:
        status_text = f"{target_date.strftime('%Y-%m-%d')} 未完成 Firstrade 周五期权筛选。"
        return [], status_text, session_out, False, ""

    rows_out: list[dict[str, T.Any]] = []
    for entry in base_rows:
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        row = {
            "symbol": symbol,
            "company": entry.get("company") or "",
            "time": entry.get("time") or "",
            "bucket": _describe_time_bucket(entry.get("time")),
            "decision_date": target_date.isoformat(),
            "timeline_date": target_date.isoformat(),
        }
        note = option_notes.get(symbol)
        if note:
            row["weekly_exp_this_fri"] = bool(note.get("weekly"))
            if note.get("expiry_type"):
                row["expiry_type"] = str(note.get("expiry_type")).upper()
        else:
            row["weekly_exp_this_fri"] = True
        raw_payload = entry.get("raw")
        if raw_payload is not None:
            row["raw"] = raw_payload
        sector = entry.get("sector") or _get_symbol_sector(symbol)
        if sector:
            row["sector"] = sector
        rows_out.append(row)

    options_applied = selection_mode in {"matched", "empty"}
    return rows_out, status_text, session_out, options_applied, ""

def start_run_logic(
    auto_intervals,
    selected_date,
    refresh_clicks,
    session_data,
    task_state,
    log_state,
):  # noqa: D401
    trigger = ctx.triggered_id
    session_state = session_data if isinstance(session_data, dict) else {}
    initial_logs = log_state if isinstance(log_state, list) else []
    logged_in = _has_valid_ft_session(session_state)
    manual_refresh = trigger == "earnings-refresh-btn"
    login_trigger = trigger == "ft-session-store"
    if trigger == "auto-run-trigger":
        if auto_intervals is None:
            return no_update, no_update, no_update, no_update
    elif trigger == "earnings-date-picker":
        if not selected_date:
            return no_update, no_update, no_update, no_update
    elif manual_refresh:
        if not refresh_clicks:
            return no_update, no_update, no_update, no_update
    elif login_trigger:
        if not session_state:
            return no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update

    if not logged_in:
        if login_trigger:
            message = "检测到无效的 Firstrade 会话，请重新登录。"
            logs = append_log(initial_logs, message, task_label="财报日程")
            return logs, message, [], no_update

        if manual_refresh or trigger == "earnings-date-picker":
            message = "尚未登录 Firstrade，请先在连接页登录后再刷新财报列表。"
            logs = append_log(initial_logs, message, task_label="财报日程")
            return logs, message, [], no_update

        return no_update, no_update, no_update, no_update

    target_date = _coerce_date(selected_date) or us_eastern_today()
    today_limit = us_eastern_today()
    max_allowed = today_limit + dt.timedelta(days=14)
    if target_date > max_allowed:
        target_date = max_allowed
    min_allowed = MIN_PICKER_DATE
    if target_date < min_allowed:
        target_date = min_allowed

    adjusted_date, weekend_shifted = _shift_weekend_to_monday(target_date)
    if adjusted_date > max_allowed:
        adjusted_date = max_allowed
        weekend_shifted = False
    if adjusted_date < min_allowed:
        adjusted_date = min_allowed
        weekend_shifted = False

    if weekend_shifted and adjusted_date != target_date:
        weekend_note = (
            f"选择的日期为周末，已自动切换至 {adjusted_date.strftime('%Y-%m-%d')}（周一）。"
        )
    else:
        weekend_note = ""
    target_date = adjusted_date

    target_date_iso = target_date.isoformat()

    if login_trigger:
        timeline_waiting = []
        for cfg in PREDICTION_TIMELINES:
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            offset_days = _resolve_timeline_offset_days(cfg)
            future_date = target_date + dt.timedelta(days=offset_days)
            detail_text = (
                f"等待预测任务 -> 将抓取 {future_date} 的财报列表并执行 {offset_label}"
            )
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
        status_message = (
            "已登录 Firstrade，预测任务将依次拉取各决策日财报并开始运行。"
        )
        logs_list: list[str] = initial_logs
        if weekend_note:
            logs_list = append_log(logs_list, weekend_note, task_label="财报日程")
        logs = append_log(logs_list, status_message, task_label="财报日程")
        return (
            logs,
            status_message,
            [],
            tasks,
        )

    cache_entry = _get_cached_earnings(target_date)
    cached_rows: list[dict[str, T.Any]] = []
    cached_status = ""
    if isinstance(cache_entry, dict):
        payload = cache_entry.get("rowData")
        if isinstance(payload, list):
            cached_rows = payload
        cached_status = str(cache_entry.get("status") or "")

    bypass_cache = manual_refresh

    if cached_rows and not bypass_cache:
        logs_list: list[str] = initial_logs
        if weekend_note:
            logs_list = append_log(logs_list, weekend_note, task_label="财报日程")
        logs = append_log(
            logs_list,
            f"命中 {target_date} 的本地缓存。",
            task_label="财报日程",
        )
        extra = "（来自本地缓存，无需重新请求。）"
        status_out = (cached_status + "\n" + extra) if cached_status else extra
        row_data = list(cached_rows)
        archive_entry = _get_prediction_archive(target_date)
        timeline_statuses = _extract_timeline_statuses(archive_entry)
        timeline_updates = []
        for cfg in PREDICTION_TIMELINES:
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            timeline_key = str(cfg.get("key") or "")
            status_info = timeline_statuses.get(timeline_key)
            if status_info:
                state = str(status_info.get("state") or "").lower()
                detail_core = status_info.get("detail") or ""
                if state == "completed":
                    status_value = "已完成"
                    detail_core = detail_core or f"{offset_label} 预测已完成"
                elif state == "empty":
                    status_value = "无数据"
                    detail_core = detail_core or f"{offset_label} 无可用预测"
                elif state in {"failed", "error"}:
                    status_value = "失败"
                    detail_core = detail_core or f"{offset_label} 上次执行失败"
                else:
                    status_value = "等待"
                    detail_core = detail_core or f"{offset_label} 预测待运行"
            else:
                status_value = "等待"
                detail_core = f"{offset_label} 预测待运行"
            timeline_updates.append(
                {
                    "id": task_id,
                    "name": name,
                    "status": status_value,
                    "detail": f"缓存数据 -> {detail_core}",
                }
            )

        tasks = _merge_task_updates(
            task_state,
            timeline_updates,
            target_date=target_date_iso,
        )
        return logs, status_out, row_data, tasks

    if cached_rows and bypass_cache:
        status_lines: list[str] = []
        logs_list: list[str] = initial_logs
        if weekend_note:
            logs_list = append_log(logs_list, weekend_note, task_label="财报日程")
            status_lines.append(weekend_note)
        manual_note = (
            "手动刷新请求已记录，但预测任务尚未启动。"
            "请在预测页开始任务以重新获取财报列表。"
        )
        status_lines.append(manual_note)
        if cached_status:
            cached_line = f"当前显示最近的缓存结果：{cached_status}"
            status_lines.append(cached_line)
        status_message = "\n".join(line for line in status_lines if line)
        logs = append_log(logs_list, manual_note, task_label="财报日程")
        return logs, status_message, list(cached_rows), no_update

    status_lines: list[str] = []
    logs_list: list[str] = initial_logs
    if weekend_note:
        logs_list = append_log(logs_list, weekend_note, task_label="财报日程")
        status_lines.append(weekend_note)

    if manual_refresh:
        waiting_message = (
            "手动刷新请求已记录，但预测任务尚未启动。"
            "请在预测页开始任务后自动获取财报列表。"
        )
    else:
        waiting_message = (
            "预测任务尚未启动。将在预测页开始任务后自动获取"
            f" {target_date.strftime('%Y-%m-%d')} 的财报列表。"
        )

    status_lines.append(waiting_message)
    status_message = "\n".join(line for line in status_lines if line)
    logs = append_log(logs_list, waiting_message, task_label="财报日程")

    timeline_waiting = []
    for idx, cfg in enumerate(PREDICTION_TIMELINES, start=1):
        task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
        offset_days = _resolve_timeline_offset_days(cfg)
        future_date = target_date + dt.timedelta(days=offset_days)
        future_label = future_date.strftime("%Y-%m-%d")
        if idx == 1:
            detail_text = (
                "等待预测任务 -> 将在任务启动后抓取 "
                f"{future_label} 的财报列表并执行 {offset_label}"
            )
        else:
            detail_text = (
                "等待预测任务 -> 前序任务完成后执行 "
                f"{offset_label}（财报日 {future_label}）"
            )
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
    table_out = list(cached_rows) if manual_refresh and cached_rows else []
    return logs, status_message, table_out, tasks



def update_predictions_logic(
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
    initial_snapshot = RL_MANAGER.snapshot() if RL_MANAGER is not None else None
    triggered = ctx.triggered_id
    if triggered == "table.selectedRows" and not selected_rows:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    valid_triggers = {"table.selectedRows", "table.rowData", "ft-session-store"}
    if triggered not in valid_triggers:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    session_data = session_state if isinstance(session_state, dict) else {}
    logged_in = _has_valid_ft_session(session_data)
    username = (username or "").strip()
    password = (password or "").strip()
    twofa = (twofa or "").strip()

    target_date = _coerce_date(picker_date) or us_eastern_today()
    adjusted_target, weekend_shifted = _shift_weekend_to_monday(target_date)
    weekend_note = ""
    if weekend_shifted and adjusted_target != target_date:
        weekend_note = (
            f"选择的日期为周末，已自动切换至 {adjusted_target.strftime('%Y-%m-%d')}（周一）。"
        )
    target_date = adjusted_target

    initial_logs = log_state if isinstance(log_state, list) else []
    log_entries = initial_logs

    existing_run_id_str = str(existing_run_id or "") if existing_run_id else ""
    if existing_run_id_str:
        existing_state = _get_prediction_run_state(existing_run_id_str)
        if existing_state and not existing_state.get("completed", False):
            status_message = str(existing_state.get("status") or "预测任务正在运行，请稍候……")
            status_out = status_message if status_message else no_update
            log_out = no_update
            return (
                no_update,
                status_out,
                no_update,
                no_update,
                log_out,
                no_update,
                no_update,
            )

    if not logged_in:
        if triggered == "ft-session-store":
            message = "检测到无效的 Firstrade 会话，请重新登录后再运行预测任务。"
            log_entries = append_log(log_entries, message, task_label="预测任务")
            log_output = log_entries if log_entries != initial_logs else no_update
            return (
                no_update,
                message,
                no_update,
                no_update,
                log_output,
                no_update,
                True,
            )
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    base_task_state = copy.deepcopy(task_state) if isinstance(task_state, dict) else {"tasks": []}
    archive_entry = _get_prediction_archive(target_date)
    timeline_statuses = _extract_timeline_statuses(archive_entry)
    existing_updates = (
        _build_task_updates_from_statuses(target_date, timeline_statuses)
        if timeline_statuses
        else []
    )
    if existing_updates:
        base_task_state = _merge_task_updates(
            base_task_state,
            existing_updates,
            target_date=target_date.isoformat(),
        )

    pending_keys: list[str] = []
    for cfg in PREDICTION_TIMELINES:
        timeline_key = str(cfg.get("key") or "")
        if not timeline_key:
            continue
        status_info = timeline_statuses.get(timeline_key)
        if _timeline_state_is_done(status_info):
            continue
        pending_keys.append(timeline_key)

    if weekend_note:
        log_entries = append_log(log_entries, weekend_note, task_label="预测任务")

    if not pending_keys:
        summary_message = "今日预测已完成。"
        if isinstance(archive_entry, dict):
            summary_message = str(archive_entry.get("status") or summary_message)
        status_lines = []
        if weekend_note:
            status_lines.append(weekend_note)
        status_lines.append(summary_message)
        status_message = "\n".join(line for line in status_lines if line)
        log_entries = append_log(log_entries, summary_message, task_label="预测任务")

        store_payload = _build_prediction_store_from_archive(archive_entry, initial_snapshot)
        task_out = base_task_state if base_task_state != task_state else no_update
        log_output = log_entries if log_entries != initial_logs else no_update
        store_out = store_payload if store_payload is not None else no_update

        return (
            store_out,
            status_message,
            no_update,
            task_out,
            log_output,
            None,
            True,
        )

    run_id = uuid.uuid4().hex

    status_lines = []
    if weekend_note:
        status_lines.append(weekend_note)
    status_lines.append(f"正在准备 {target_date.strftime('%Y-%m-%d')} 的预测任务……")
    status_message = "\n".join(status_lines)

    _init_prediction_run_state(run_id, target_date, log_entries, base_task_state, status_message)

    thread = threading.Thread(
        target=_prediction_thread_worker,
        args=(
            run_id,
            triggered,
            copy.deepcopy(session_data),
            username,
            password,
            twofa,
            target_date,
            initial_snapshot,
            copy.deepcopy(base_task_state),
            list(log_entries),
            copy.deepcopy(archive_entry) if isinstance(archive_entry, dict) else None,
            tuple(pending_keys),
        ),
        daemon=True,
    )
    thread.start()

    log_output = log_entries if log_entries != initial_logs else no_update

    return (
        no_update,
        status_message,
        no_update,
        no_update,
        log_output,
        run_id,
        False,
    )


def _prediction_thread_worker(
    run_id: str,
    triggered: str | None,
    session_data: dict[str, T.Any],
    username: str,
    password: str,
    twofa: str,
    target_date: dt.date,
    initial_snapshot: T.Any,
    initial_task_state: dict[str, T.Any],
    initial_logs: list[str],
    archive_entry: dict[str, T.Any] | None,
    pending_timelines: tuple[str, ...] | None,
) -> None:
    log_entries = list(initial_logs or [])
    task_state_local = (
        copy.deepcopy(initial_task_state)
        if isinstance(initial_task_state, dict)
        else {"tasks": []}
    )
    session_local = copy.deepcopy(session_data) if isinstance(session_data, dict) else {}
    target_date_iso = target_date.isoformat()

    archive_snapshot = archive_entry if isinstance(archive_entry, dict) else {}
    existing_statuses = _extract_timeline_statuses(archive_snapshot)
    pending_set: set[str] = {str(key) for key in (pending_timelines or ()) if key}
    if not pending_set:
        for cfg in PREDICTION_TIMELINES:
            key = str(cfg.get("key") or "")
            if key and not _timeline_state_is_done(existing_statuses.get(key)):
                pending_set.add(key)

    try:
        pending_preview = ", ".join(sorted(pending_set)) or "无待处理时点"
    except TypeError:  # pragma: no cover - defensive
        pending_preview = "无待处理时点"

    row_groups_existing = _group_rowdata_by_timeline(archive_snapshot.get("rowData"))
    result_groups_existing = _group_results_by_timeline(archive_snapshot.get("results"))
    timeline_sources_snapshot = (
        archive_snapshot.get("timeline_sources")
        if isinstance(archive_snapshot.get("timeline_sources"), dict)
        else {}
    )

    result_lookup: dict[tuple[str, str], dict[str, T.Any]] = {}
    for timeline_key_init, entries in result_groups_existing.items():
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            result_lookup[(symbol, timeline_key_init)] = dict(entry)

    missing_existing: set[str] = set()
    missing_reason_lookup: dict[str, str] = {}
    raw_missing = archive_snapshot.get("missing") if isinstance(archive_snapshot, dict) else None
    if isinstance(raw_missing, list):
        for item in raw_missing:
            if item is None:
                continue
            key = str(item)
            missing_existing.add(key)
            missing_reason_lookup[key] = "历史存档标记为缺失"

    errors_existing: set[str] = set()
    raw_errors = archive_snapshot.get("errors") if isinstance(archive_snapshot, dict) else None
    if isinstance(raw_errors, list):
        for item in raw_errors:
            if item is None:
                continue
            errors_existing.add(str(item))

    def _emit(message: str, task_label: str | None = None) -> None:
        nonlocal log_entries
        if not message:
            return
        log_entries = append_log(log_entries, message, task_label=task_label)
        _update_prediction_run_state(run_id, logs=log_entries, status=message)

    def _set_tasks(updated: dict[str, T.Any]) -> None:
        nonlocal task_state_local
        task_state_local = updated
        _update_prediction_run_state(run_id, tasks=task_state_local)

    run_identifier = uuid.uuid4().hex

    try:
        _emit(
            "预测线程已启动，准备执行关键步骤……",
            task_label="预测任务",
        )
        _emit(
            f"目标日期：{target_date_iso}，待处理时点：{pending_preview}",
            task_label="预测任务",
        )
        if triggered == "ft-session-store":
            summary_msg = (
                f"{target_date.strftime('%m月%d日')} 的预测序列已启动，共 {len(PREDICTION_TIMELINES)} 个时点。"
            )
            _emit(summary_msg, task_label="预测任务")


        timeline_dates: dict[str, dt.date] = {}
        timeline_summary: dict[str, dict[str, T.Any]] = {}
        timeline_reports: dict[str, str] = {}
        task_lookup: dict[str, tuple[str, str, str]] = {}
        progress_tracker: dict[str, dict[str, int]] = {}
        timeline_symbol_counts: dict[str, int] = {}
        timeline_metadata_map: dict[str, dict[str, dict[str, T.Any]]] = {}
        timeline_status_notes: dict[str, tuple[str, str]] = {}
        timeline_final_statuses: dict[str, dict[str, T.Any]] = {}
        aggregated_row_data: list[dict[str, T.Any]] = []
        all_missing_set: set[str] = set(missing_existing)
        all_errors_set: set[str] = set(errors_existing)

        def _progress(symbol: str, timeline: dict[str, T.Any], stage: str, detail: str | None) -> None:
            nonlocal log_entries
            label = str(timeline.get("label") or timeline.get("key") or "未知时点")
            key = str(timeline.get("key") or label)
            timeline_date_value = timeline_dates.get(key)
            date_str = timeline_date_value.isoformat() if isinstance(timeline_date_value, dt.date) else "-"
            meta_entry = (timeline_metadata_map.get(key) or {}).get(symbol, {})
            bucket_label = str(meta_entry.get("bucket") or "")
            bucket_part = f"（{bucket_label}）" if bucket_label else ""
            message: str | None = None
            if stage == "start":
                tracker = progress_tracker.setdefault(
                    key,
                    {
                        "total": timeline_symbol_counts.get(key, 0),
                        "completed": 0,
                        "processed": 0,
                    },
                )
                current_index = tracker.get("processed", 0) + 1
                message = (
                    f"准备标的 #{current_index}：{symbol}{bucket_part}｜决策日 {date_str}"
                )
            elif stage == "success":
                timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
                timeline_summary[key]["success"] = timeline_summary[key].get("success", 0) + 1
                message = f"完成 {symbol} 的 {label} 预测{bucket_part}。"
            elif stage == "missing":
                timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
                timeline_summary[key]["missing"] = timeline_summary[key].get("missing", 0) + 1
                if detail:
                    detail_note = f"（{detail}）"
                    message = f"{symbol} 的 {label} 输入缺失，跳过该时点{bucket_part}{detail_note}。"
                else:
                    message = None
            elif stage == "error":
                timeline_summary.setdefault(key, {"label": label, "success": 0, "missing": 0, "error": 0})
                timeline_summary[key]["error"] = timeline_summary[key].get("error", 0) + 1
                detail_text = f"：{detail}" if detail else ""
                message = f"{symbol} 的 {label} 预测失败{bucket_part}{detail_text}。"
            if message:
                _emit(message, task_label=label)

            if stage in {"success", "missing", "error"}:
                total_for_task = timeline_symbol_counts.get(key, 0)
                tracker = progress_tracker.setdefault(
                    key,
                    {"total": total_for_task, "completed": 0, "processed": 0},
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
                    prefix = f"{lookup[2]}｜决策日 {date_str}"
                    update_payload = {
                        "id": lookup[0],
                        "name": lookup[1],
                        "detail": f"{prefix}：" + "，".join(progress_parts),
                        "total_symbols": tracker["total"],
                        "completed_symbols": tracker["completed"],
                        "processed_symbols": tracker["processed"],
                    }
                    updated_state = _merge_task_updates(
                        task_state_local,
                        [update_payload],
                        target_date=target_date_iso,
                    )
                    _set_tasks(updated_state)

        for cfg in PREDICTION_TIMELINES:
            task_id, name, offset_label = _describe_timeline_task(target_date, cfg)
            timeline_key = str(cfg.get("key") or "")
            offset_days = _resolve_timeline_offset_days(cfg)
            timeline_date = target_date + dt.timedelta(days=offset_days)
            timeline_dates[timeline_key] = timeline_date
            label = str(cfg.get("label") or offset_label)
            task_lookup[timeline_key] = (task_id, name, offset_label)
            timeline_metadata_map[timeline_key] = {}
            timeline_symbol_counts[timeline_key] = 0
            timeline_status_notes.setdefault(timeline_key, ("", ""))
            timeline_final_statuses.setdefault(timeline_key, {})

            timeline_date_str = timeline_date.isoformat()
            existing_meta_map = row_groups_existing.get(timeline_key, {})
            should_process = timeline_key in pending_set

            if should_process:
                _emit(
                    f"开始准备 {label}（决策日 {timeline_date_str}）的预测任务。",
                    task_label=label,
                )
            else:
                _emit(
                    f"{label}（决策日 {timeline_date_str}）已存在记录，将检查是否需要复用。",
                    task_label=label,
                )

            if not should_process:
                meta_map = {symbol: dict(data) for symbol, data in existing_meta_map.items()}
                timeline_metadata_map[timeline_key] = meta_map
                timeline_symbol_counts[timeline_key] = len(meta_map)
                state_info = existing_statuses.get(timeline_key) or {}
                state_value = str(
                    state_info.get("state") or ("completed" if meta_map else "empty")
                ).lower()
                if state_value not in {"completed", "empty"}:
                    state_value = "completed" if meta_map else "empty"
                detail_body = state_info.get("detail") or (
                    "存档中已有预测结果" if state_value == "completed" else "存档记录为空结果"
                )
                detail_text = f"{offset_label}｜决策日 {timeline_date_str}：{detail_body}"
                status_flag = "ok" if state_value == "completed" else "empty"
                timeline_status_notes[timeline_key] = (status_flag, detail_body)
                timeline_reports[timeline_key] = detail_body
                source_date = state_info.get("source_date") or str(
                    timeline_sources_snapshot.get(timeline_key) or timeline_date_str
                )
                timeline_final_statuses[timeline_key] = {
                    "state": state_value,
                    "detail": detail_body,
                    "symbol_count": len(meta_map),
                    "source_date": source_date,
                    "updated_at": state_info.get("updated_at")
                    or dt.datetime.now(US_EASTERN).isoformat(),
                }
                status_label = "已完成" if state_value == "completed" else "无数据"
                end_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
                update_state = _merge_task_updates(
                    task_state_local,
                    [
                        {
                            "id": task_id,
                            "name": name,
                            "status": status_label,
                            "detail": detail_text,
                            "end_time": end_ts,
                            "total_symbols": len(meta_map),
                            "completed_symbols": len(meta_map) if state_value == "completed" else 0,
                            "processed_symbols": len(meta_map),
                        }
                    ],
                    target_date=target_date_iso,
                )
                _set_tasks(update_state)
                message = (
                    f"{name}（决策日 {timeline_date_str}）已在存档中完成，跳过重新计算。"
                    if state_value == "completed"
                    else f"{name}（决策日 {timeline_date_str}）存档为空，跳过重新计算。"
                )
                _emit(message, task_label=label)
                continue

            for result_key in list(result_lookup.keys()):
                if result_key[1] == timeline_key:
                    result_lookup.pop(result_key, None)

            fetch_update = _merge_task_updates(
                task_state_local,
                [
                    {
                        "id": task_id,
                        "name": name,
                        "status": "进行中",
                        "detail": f"{offset_label}｜决策日 {timeline_date_str}：正在抓取财报列表",
                        "start_time": dt.datetime.now(US_EASTERN).strftime("%H:%M:%S"),
                        "end_time": "",
                    }
                ],
                target_date=target_date_iso,
            )
            _set_tasks(fetch_update)
            _emit(f"{name}（决策日 {timeline_date_str}）已开始抓取财报列表。", task_label=label)

            cached_entry = _get_cached_earnings(timeline_date)
            rows: list[dict[str, T.Any]] = []
            status_text = ""
            if isinstance(cached_entry, dict):
                payload = cached_entry.get("rowData")
                if isinstance(payload, list):
                    rows = payload
                status_text = str(cached_entry.get("status") or "")
                if rows:
                    timeline_status_notes[timeline_key] = (
                        "ok",
                        status_text or f"命中缓存，共 {len(rows)} 个标的。",
                    )
                    _emit(
                        f"{label}（{timeline_date_str}）命中缓存，共 {len(rows)} 个标的。",
                        task_label=label,
                    )

            if not rows:
                _emit(
                    f"未找到 {timeline_date} 的筛选财报列表（{label}），开始自动刷新。",
                    task_label=label,
                )
                fetched_rows, status_text, session_out, options_applied, error_message = _prepare_earnings_dataset(
                    timeline_date,
                    session_local,
                    username,
                    password,
                    twofa,
                    logger=(lambda message, tag=label: _emit(message, task_label=tag)),
                )
                if isinstance(session_out, dict) and session_out:
                    session_local = session_out
                if error_message:
                    detail_message = f"{label} 财报列表获取失败：{error_message}"
                    timeline_status_notes[timeline_key] = ("fail", detail_message)
                    _emit(detail_message, task_label=label)
                    rows = []
                elif options_applied is not True:
                    detail_message = f"{label} 财报列表未完成 Firstrade 筛选，已跳过。"
                    timeline_status_notes[timeline_key] = ("fail", detail_message)
                    _emit(detail_message, task_label=label)
                    rows = []
                else:
                    rows = fetched_rows
                    if rows:
                        _store_cached_earnings(
                            timeline_date,
                            rows,
                            status_text or "",
                            options_filter_applied=True,
                        )
                        timeline_status_notes[timeline_key] = (
                            "ok",
                            status_text or f"刷新后共 {len(rows)} 个标的。",
                        )

            meta_map: dict[str, dict[str, T.Any]] = {}
            if rows:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    symbol = str(row.get("symbol") or "").upper()
                    if not symbol:
                        continue
                    enriched = dict(row)
                    enriched.setdefault("timeline_key", timeline_key)
                    enriched.setdefault("timeline_date", timeline_date_str)
                    meta_map[symbol] = enriched
                timeline_metadata_map[timeline_key] = meta_map
                timeline_symbol_counts[timeline_key] = len(meta_map)
                _emit(
                    f"{label}（{timeline_date_str}）共 {len(meta_map)} 个标的。",
                    task_label=label,
                )
                timeline_status_notes.setdefault(
                    timeline_key,
                    ("ok", f"共 {len(meta_map)} 个标的"),
                )
            else:
                timeline_metadata_map[timeline_key] = {}
                if timeline_status_notes.get(timeline_key, ("", ""))[0] != "fail":
                    detail_message = f"{label}（{timeline_date_str}）无可用财报标的。"
                    timeline_status_notes[timeline_key] = ("empty", detail_message)
                    _emit(detail_message, task_label=label)

            status_code, status_note = timeline_status_notes.get(timeline_key, ("empty", ""))
            symbol_count = timeline_symbol_counts.get(timeline_key, 0)

            if status_code == "fail":
                detail_text = (
                    f"{offset_label}｜决策日 {timeline_date_str}：{status_note or '财报列表获取失败'}"
                )
                end_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
                updated_state = _merge_task_updates(
                    task_state_local,
                    [
                        {
                            "id": task_id,
                            "name": name,
                            "status": "失败",
                            "detail": detail_text,
                            "end_time": end_ts,
                            "total_symbols": symbol_count,
                            "completed_symbols": 0,
                            "processed_symbols": 0,
                        }
                    ],
                    target_date=target_date_iso,
                )
                _set_tasks(updated_state)
                _emit(f"{name} 失败：{status_note}", task_label=label)
                timeline_reports[timeline_key] = detail_text.split("：", 1)[-1]
                timeline_final_statuses[timeline_key] = {
                    "state": "failed",
                    "detail": detail_text.split("：", 1)[-1],
                    "symbol_count": symbol_count,
                    "source_date": timeline_date_str,
                    "updated_at": dt.datetime.now(US_EASTERN).isoformat(),
                }
                continue

            if symbol_count <= 0:
                base_message = status_note or "无筛选标的"
                detail_text = f"{offset_label}｜决策日 {timeline_date_str}：{base_message}"
                end_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
                updated_state = _merge_task_updates(
                    task_state_local,
                    [
                        {
                            "id": task_id,
                            "name": name,
                            "status": "无数据",
                            "detail": detail_text,
                            "end_time": end_ts,
                            "total_symbols": 0,
                            "completed_symbols": 0,
                            "processed_symbols": 0,
                        }
                    ],
                    target_date=target_date_iso,
                )
                _set_tasks(updated_state)
                timeline_reports[timeline_key] = detail_text.split("：", 1)[-1]
                timeline_final_statuses[timeline_key] = {
                    "state": "empty",
                    "detail": detail_text.split("：", 1)[-1],
                    "symbol_count": 0,
                    "source_date": timeline_date_str,
                    "updated_at": dt.datetime.now(US_EASTERN).isoformat(),
                }
                continue

            progress_tracker[timeline_key] = {
                "total": symbol_count,
                "completed": 0,
                "processed": 0,
            }
            running_detail = (
                f"{offset_label}｜决策日 {timeline_date_str}：正在处理 {symbol_count} 个标的"
            )
            updated_state = _merge_task_updates(
                task_state_local,
                [
                    {
                        "id": task_id,
                        "name": name,
                        "status": "进行中",
                        "detail": running_detail,
                        "total_symbols": symbol_count,
                        "completed_symbols": 0,
                        "processed_symbols": 0,
                    }
                ],
                target_date=target_date_iso,
            )
            _set_tasks(updated_state)
            _emit(f"{name}（决策日 {timeline_date_str}）已开始。", task_label=label)

            metadata_map = timeline_metadata_map.get(timeline_key, {})
            symbols = list(metadata_map.keys())
            dci_payloads = _load_dci_payloads()
            (
                available_symbols,
                missing_symbols,
                fallback_symbols,
            ) = _summarise_dci_payloads(
                symbols,
                dci_payloads,
            )

            summary_parts: list[str] = []
            if available_symbols:
                sample = "，".join(available_symbols[:6])
                more = "……" if len(available_symbols) > 6 else ""
                summary_parts.append(
                    f"命中 {len(available_symbols)} 个标的（{sample}{more}）"
                )
            if fallback_symbols:
                sample = "，".join(fallback_symbols[:6])
                more = "……" if len(fallback_symbols) > 6 else ""
                summary_parts.append(
                    f"对 {len(fallback_symbols)} 个标的使用内置基准因子（{sample}{more}）"
                )
            if missing_symbols:
                sample = "，".join(missing_symbols[:6])
                more = "……" if len(missing_symbols) > 6 else ""
                summary_parts.append(
                    f"数据源未覆盖 {len(missing_symbols)} 个标的（{sample}{more}）"
                )

            if summary_parts:
                _emit(
                    f"已加载 {label} 的 DCI 输入：" + "；".join(summary_parts),
                    task_label=label,
                )

            (
                partial_results,
                partial_missing,
                partial_errors,
                partial_missing_detail,
            ) = _compute_dci_for_symbols(
                symbols,
                metadata_map,
                progress_callback=_progress,
                timeline_configs=[cfg],
                payloads=dci_payloads,
            )

            for entry in partial_results:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("symbol") or "").upper()
                timeline_label = str(entry.get("timeline_key") or timeline_key)
                if not symbol or not timeline_label:
                    continue
                result_lookup[(symbol, timeline_label)] = entry

            for item in partial_missing:
                if item is None:
                    continue
                all_missing_set.add(str(item))
            if isinstance(partial_missing_detail, dict):
                grouped_missing_detail: dict[str, list[str]] = {}
                for miss_key, miss_reason in partial_missing_detail.items():
                    if not miss_key:
                        continue
                    if not miss_reason:
                        continue
                    reason_text = str(miss_reason)
                    key_text = str(miss_key)
                    missing_reason_lookup[key_text] = reason_text
                    grouped_missing_detail.setdefault(reason_text, []).append(key_text)
                for reason_text, miss_entries in grouped_missing_detail.items():
                    if not miss_entries:
                        continue
                    sample_text = "；".join(miss_entries[:5])
                    if len(miss_entries) > 5:
                        sample_text += "……"
                    _emit(
                        f"{label} 缺数据原因（{reason_text}）：{len(miss_entries)} 条（{sample_text}）",
                        task_label=label,
                    )

            for item in partial_errors:
                if item is None:
                    continue
                all_errors_set.add(str(item))

            summary = timeline_summary.get(timeline_key) or {}
            success = int(summary.get("success", 0) or 0)
            missing_cnt = int(summary.get("missing", 0) or 0)
            error_cnt = int(summary.get("error", 0) or 0)
            _emit(
                f"{label}（{timeline_date_str}）DCI 处理完成：成功 {success}，缺数据 {missing_cnt}，失败 {error_cnt}。",
                task_label=label,
            )
            parts: list[str] = []
            if success:
                parts.append(f"完成{success}次")
            if missing_cnt:
                parts.append(f"缺数据{missing_cnt}次")
            if error_cnt:
                parts.append(f"失败{error_cnt}次")
            detail_text = f"{offset_label}｜决策日 {timeline_date_str}：" + ("，".join(parts) if parts else "无数据")
            tracker_final = progress_tracker.get(timeline_key) or {}
            total_symbols_for_task = tracker_final.get("total", symbol_count)
            completed_for_task = tracker_final.get("completed", success)
            processed_for_task = tracker_final.get("processed", success + missing_cnt + error_cnt)
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
            detail_body = detail_text.split("：", 1)[-1]
            timeline_reports[timeline_key] = detail_body

            if success:
                final_state = "completed"
            elif error_cnt:
                final_state = "failed"
            else:
                final_state = "empty"

            status_value = {
                "completed": "已完成",
                "failed": "失败",
                "empty": "无数据",
            }[final_state]

            timeline_final_statuses[timeline_key] = {
                "state": final_state,
                "detail": detail_body,
                "symbol_count": total_symbols_for_task,
                "source_date": timeline_date_str,
                "updated_at": dt.datetime.now(US_EASTERN).isoformat(),
            }

            end_ts = dt.datetime.now(US_EASTERN).strftime("%H:%M:%S")
            final_update = _merge_task_updates(
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
            _set_tasks(final_update)
            _emit(f"{name} 完成：{detail_text}", task_label=label)
        aggregated_row_data = []
        for meta_map in timeline_metadata_map.values():
            aggregated_row_data.extend(meta_map.values())

        results = sorted(
            list(result_lookup.values()),
            key=lambda row: (row.get("symbol", ""), row.get("lookback_days", 0)),
        )
        missing = sorted(all_missing_set)
        errors = sorted(all_errors_set)

        snapshot = RL_MANAGER.snapshot() if RL_MANAGER is not None else initial_snapshot

        lines: list[str] = []
        if results:
            lines.append(f"已生成 {len(results)} 个预测结果（每个标的最多 {len(PREDICTION_TIMELINES)} 个时点）。")
        else:
            lines.append("未生成新的预测结果。")

        if missing:
            preview_missing = "；".join(missing[:5])
            if len(missing) > 5:
                preview_missing += "……"
            lines.append(f"缺少数据的条目：{preview_missing}")
            grouped_missing: dict[str, list[str]] = {}
            for miss_key in missing:
                reason_text = missing_reason_lookup.get(miss_key)
                if not reason_text:
                    continue
                grouped_missing.setdefault(reason_text, []).append(miss_key)
            for reason_text, miss_entries in grouped_missing.items():
                sample_text = "；".join(miss_entries[:5])
                if len(miss_entries) > 5:
                    sample_text += "……"
                lines.append(f"{reason_text}：{sample_text}")

        if errors:
            preview_errors = "；".join(errors[:5])
            if len(errors) > 5:
                preview_errors += "……"
            lines.append(f"出现错误的条目：{preview_errors}")

        summary_texts: list[str] = []
        if timeline_reports:
            for cfg in PREDICTION_TIMELINES:
                key = str(cfg.get("key") or "")
                label = str(cfg.get("label") or key)
                date_value = timeline_dates.get(key)
                date_str = date_value.isoformat() if isinstance(date_value, dt.date) else "-"
                report = timeline_reports.get(key)
                if report:
                    lines.append(f"- {label}（{date_str}）：{report}")
                    summary_texts.append(f"{label} -> {report}")
        if summary_texts:
            _emit("时点统计：" + "；".join(summary_texts), task_label="预测汇总")

        message = "\n".join(lines)

        _emit(
            f"预测汇总：结果 {len(results)} 条，缺数据 {len(missing)} 条，错误 {len(errors)} 条。",
            task_label="预测任务",
        )

        if lines:
            _emit("预测任务总结：", task_label="预测汇总")
            for line in lines:
                if line:
                    _emit(line, task_label="预测汇总")

        timeline_sources = {
            key: value.isoformat() for key, value in timeline_dates.items() if isinstance(value, dt.date)
        }

        _emit("正在写入预测结果到存档……", task_label="预测任务")
        _store_prediction_results(
            target_date,
            aggregated_row_data,
            results,
            message or "",
            timeline_sources,
            run_identifier=run_identifier,
            missing=missing,
            errors=errors,
            timeline_statuses=timeline_final_statuses,
        )
        _emit("预测存档写入完成。", task_label="预测任务")

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

        store_payload = {
            "results": results,
            "missing": missing,
            "errors": errors,
            "rl_snapshot": snapshot,
        }

        _update_prediction_run_state(
            run_id,
            store_data=store_payload,
            rl_snapshot=snapshot,
            logs=log_entries,
            tasks=task_state_local,
            status=message or "预测任务已完成。",
            completed=True,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        error_message = f"预测任务失败：{exc}"
        _emit(error_message, task_label="预测任务")
        tb_text = traceback.format_exc()
        if tb_text:
            for line in tb_text.strip().splitlines():
                _emit(line, task_label="预测任务")
        fallback_store = {
            "results": [],
            "missing": [],
            "errors": [str(exc)],
            "rl_snapshot": initial_snapshot,
        }
        _update_prediction_run_state(
            run_id,
            store_data=fallback_store,
            rl_snapshot=initial_snapshot,
            logs=log_entries,
            tasks=task_state_local,
            status=error_message,
            completed=True,
            error=str(exc),
        )


def poll_prediction_run_logic(
    n_intervals,
    run_id,
    existing_store,
    existing_status,
    existing_snapshot,
    existing_task_state,
    existing_logs,
):  # noqa: D401
    del n_intervals

    if not run_id:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            no_update,
        )

    state = _get_prediction_run_state(str(run_id))
    if not state:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            None,
        )

    logs = state.get("logs")
    store_data = state.get("store_data")
    status = state.get("status")
    snapshot = state.get("rl_snapshot")
    tasks = state.get("tasks")
    completed = bool(state.get("completed"))

    log_out = logs if isinstance(logs, list) and logs != existing_logs else no_update
    store_out = store_data if isinstance(store_data, dict) and store_data != existing_store else no_update
    status_out = status if isinstance(status, str) and status != existing_status else no_update
    snapshot_out = (
        snapshot if snapshot is not None and snapshot != existing_snapshot else no_update
    )
    task_out = tasks if isinstance(tasks, dict) and tasks != existing_task_state else no_update

    disable_flag = bool(completed)
    run_id_out = None if completed else no_update

    if completed:
        _clear_prediction_run_state(str(run_id))

    return (
        store_out,
        status_out,
        snapshot_out,
        task_out,
        log_out,
        disable_flag,
        run_id_out,
    )

def render_prediction_table_logic(store_data):  # noqa: D401
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

def _is_trading_day(date_value: dt.date) -> bool:
    """Return whether the given date is a US trading day (Mon-Fri)."""

    return date_value.weekday() < 5


def _has_decision_day_predictions(archive_entry: dict[str, T.Any] | None) -> bool:
    """Return True if the archive entry contains decision-day predictions."""

    if not isinstance(archive_entry, dict):
        return False

    raw_results = archive_entry.get("results")
    if not isinstance(raw_results, list):
        return False
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        timeline_key = str(result.get("timeline_key") or "")
        lookback = result.get("lookback_days")
        if timeline_key == "decision_day" or (isinstance(lookback, (int, float)) and int(lookback) == 0):
            return True
    return False


def _has_evaluation_record(archive_entry: dict[str, T.Any] | None) -> bool:
    """Return True if the archive entry already carries a validation record."""

    if not isinstance(archive_entry, dict):
        return False

    evaluation = archive_entry.get("evaluation")
    if not isinstance(evaluation, dict):
        return False
    summary = evaluation.get("summary")
    if isinstance(summary, dict) and summary.get("checked_at"):
        return True
    items = evaluation.get("items")
    return isinstance(items, list) and bool(items)


def _find_validation_target(start_date: dt.date) -> tuple[dt.date | None, bool]:
    """Locate the most recent decision day that requires validation.

    The search skips the immediate previous trading day (``start_date``) because
    its T+1 results are not yet available during the morning run. The returned
    tuple consists of ``(date, is_backfilled)`` where ``is_backfilled`` indicates
    whether the located date is older than ``start_date``.
    """

    candidate = start_date
    fallback: dt.date | None = None
    skipped_start = False
    for _ in range(120):  # limit search horizon to avoid infinite loops
        if not skipped_start and candidate == start_date:
            skipped_start = True
            candidate = previous_trading_day(candidate)
            continue

        if not _is_trading_day(candidate):
            candidate -= dt.timedelta(days=1)
            continue

        entry = _get_prediction_archive(candidate)
        if not _has_decision_day_predictions(entry):
            candidate = previous_trading_day(candidate)
            continue

        if not _has_evaluation_record(entry):
            return candidate, candidate < start_date

        if fallback is None:
            fallback = candidate
        candidate = previous_trading_day(candidate)

    if fallback is not None:
        return fallback, fallback < start_date
    return None, False


def _resolve_tradingview_exchange(
    symbol: str, row_meta: dict[str, T.Any] | None = None
) -> str | None:
    """Guess the TradingView exchange code for the given symbol."""

    raw_exchange = ""
    if isinstance(row_meta, dict):
        raw_exchange = str(row_meta.get("exchange") or "")
        if not raw_exchange:
            raw_payload = row_meta.get("raw")
            if isinstance(raw_payload, dict):
                raw_exchange = str(
                    raw_payload.get("exchange")
                    or raw_payload.get("exchangeName")
                    or raw_payload.get("exchange_code")
                    or ""
                )

    if not raw_exchange:
        metadata = _get_symbol_metadata(symbol)
        if isinstance(metadata, dict):
            raw_exchange = str(metadata.get("exchange") or "")

    raw_upper = raw_exchange.upper().strip()
    for key, mapped in TV_EXCHANGE_ALIASES.items():
        if key in raw_upper:
            return mapped

    if TV_DEFAULT_EXCHANGE:
        return TV_DEFAULT_EXCHANGE

    return raw_upper or None


def _compute_trade_window(decision_day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Return the entry (15:00) and exit (次日09:40) timestamps in US Eastern."""

    entry_time = dt.datetime.combine(decision_day, dt.time(15, 0), tzinfo=US_EASTERN)
    exit_day = next_trading_day(decision_day)
    exit_time = dt.datetime.combine(exit_day, dt.time(9, 40), tzinfo=US_EASTERN)
    return entry_time, exit_time


def _round_option_strike(price: float) -> float:
    """Round the underlying price to a representative ATM strike."""

    if not math.isfinite(price) or price <= 0:
        return 0.0
    if price < 25:
        step = 0.5
    elif price < 200:
        step = 1.0
    elif price < 500:
        step = 5.0
    else:
        step = 10.0
    return round(price / step) * step


def _determine_option_expiry(
    decision_day: dt.date, row_meta: dict[str, T.Any] | None = None
) -> dt.date:
    """Choose an expiry date for the ATM option used in validation."""

    candidate: dt.date | None = None

    if isinstance(row_meta, dict):
        for key in ("option_expiry", "expiry_date", "expiration", "expiration_date"):
            value = row_meta.get(key)
            if isinstance(value, str):
                try:
                    candidate = dt.date.fromisoformat(value[:10])
                except ValueError:
                    continue
                else:
                    break
        if candidate is None:
            raw_payload = row_meta.get("raw")
            if isinstance(raw_payload, dict):
                for key in ("expiryDate", "expirationDate", "expDate"):
                    value = raw_payload.get(key)
                    if isinstance(value, str):
                        try:
                            candidate = dt.date.fromisoformat(value[:10])
                        except ValueError:
                            continue
                        else:
                            break

    if candidate is None and isinstance(row_meta, dict):
        weekly_flag = row_meta.get("weekly_exp_this_fri")
        if weekly_flag in {True, "True", "true", 1}:
            candidate = this_friday(decision_day)

    if candidate is None:
        candidate = this_friday(decision_day)

    return candidate


def _build_option_symbol_variants(
    symbol: str, expiry: dt.date, option_type: str, strike: float
) -> list[str]:
    """Return plausible TradingView option symbols for the given contract."""

    base = symbol.upper()
    occ_value = int(round(strike * 1000))
    occ_part = f"{occ_value:08d}"
    if math.isclose(strike, round(strike)):
        strike_simple = f"{int(round(strike))}"
    else:
        strike_simple = f"{strike:.2f}".rstrip("0").rstrip(".")

    candidates = [
        f"{base}{expiry:%y%m%d}{option_type}{occ_part}",
        f"{base}{expiry:%y%m%d}{option_type}{strike_simple}",
        f"{base}{expiry:%y%m%d}{option_type}{strike_simple.replace('.', '')}",
    ]
    seen: set[str] = set()
    variants: list[str] = []
    for item in candidates:
        if item and item not in seen:
            variants.append(item)
            seen.add(item)
    return variants


def _fetch_underlying_trade_points(
    symbol: str,
    exchange: str,
    entry_time: dt.datetime,
    exit_time: dt.datetime,
) -> tuple[tv_data.PricePoint | None, tv_data.PricePoint | None, list[str]]:
    """Fetch entry/exit minute prices for the underlying symbol."""

    errors: list[str] = []
    window_start = entry_time - dt.timedelta(hours=8)
    window_end = exit_time + dt.timedelta(hours=4)

    try:
        data = tv_data.fetch_hist(
            symbol,
            exchange,
            start=window_start,
            end=window_end,
            interval=tv_data.DEFAULT_INTERVAL,
        )
    except tv_data.TVDataError as exc:
        errors.append(str(exc))
        return None, None, errors

    entry_point = tv_data.fetch_price_point(
        data,
        entry_time,
        tolerance=dt.timedelta(minutes=TV_PRICE_TOLERANCE_MINUTES),
    )
    exit_point = tv_data.fetch_price_point(
        data,
        exit_time,
        tolerance=dt.timedelta(minutes=TV_PRICE_TOLERANCE_MINUTES),
    )

    if entry_point is None:
        errors.append("缺少 15:00 的现货报价")
    if exit_point is None:
        errors.append("缺少 次日09:40 的现货报价")

    return entry_point, exit_point, errors


def _fetch_option_trade_points(
    symbol: str,
    predicted_label: str,
    entry_time: dt.datetime,
    exit_time: dt.datetime,
    *,
    row_meta: dict[str, T.Any] | None,
    entry_price: float | None,
) -> tuple[dict[str, T.Any] | None, list[str]]:
    """Fetch entry/exit prices for the ATM option contract."""

    errors: list[str] = []
    if entry_price is None or entry_price <= 0:
        return None, ["无法根据现货价格确定 ATM 行权价"]

    option_type = "C" if predicted_label == "多" else "P"
    expiry = _determine_option_expiry(entry_time.date(), row_meta)
    strike = _round_option_strike(entry_price)
    if strike <= 0:
        return None, ["未能确定有效的行权价"]

    exchange = TV_OPTION_EXCHANGE or _resolve_tradingview_exchange(symbol, row_meta)
    if not exchange:
        return None, ["无法确定期权交易所"]

    variants = _build_option_symbol_variants(symbol, expiry, option_type, strike)
    window_start = entry_time - dt.timedelta(hours=2)
    window_end = exit_time + dt.timedelta(hours=2)

    for candidate in variants:
        try:
            data = tv_data.fetch_hist(
                candidate,
                exchange,
                start=window_start,
                end=window_end,
                interval=tv_data.DEFAULT_INTERVAL,
            )
        except tv_data.TVDataError as exc:
            errors.append(f"{candidate}@{exchange}: {exc}")
            continue

        entry_point = tv_data.fetch_price_point(
            data,
            entry_time,
            tolerance=dt.timedelta(minutes=TV_OPTION_TOLERANCE_MINUTES),
        )
        exit_point = tv_data.fetch_price_point(
            data,
            exit_time,
            tolerance=dt.timedelta(minutes=TV_OPTION_TOLERANCE_MINUTES),
        )

        if entry_point and exit_point:
            return (
                {
                    "symbol": candidate,
                    "exchange": exchange,
                    "strike": strike,
                    "expiry": expiry,
                    "entry": entry_point,
                    "exit": exit_point,
                },
                errors,
            )

        errors.append(f"{candidate}@{exchange}: 缺少目标时间段报价")

    return None, errors


def _evaluate_trade_window(
    symbol: str,
    decision_day: dt.date,
    predicted_label: str,
    row_meta: dict[str, T.Any] | None,
) -> tuple[dict[str, T.Any] | None, list[str]]:
    """Gather intraday prices for underlying and ATM option via tvDatafeed."""

    if predicted_label == "放弃":
        return None, []

    exchange = _resolve_tradingview_exchange(symbol, row_meta)
    if not exchange:
        return None, ["无法确定股票所属交易所"]

    entry_time, exit_time = _compute_trade_window(decision_day)
    entry_point, exit_point, base_errors = _fetch_underlying_trade_points(
        symbol, exchange, entry_time, exit_time
    )

    payload: dict[str, T.Any] = {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "underlying_entry": entry_point,
        "underlying_exit": exit_point,
        "underlying_exchange": exchange,
    }

    errors = list(base_errors)

    if entry_point is None or exit_point is None:
        return payload, errors

    option_data, option_errors = _fetch_option_trade_points(
        symbol,
        predicted_label,
        entry_time,
        exit_time,
        row_meta=row_meta,
        entry_price=entry_point.price,
    )
    payload["option"] = option_data
    errors.extend(option_errors)

    return payload, errors


def auto_evaluate_predictions_logic(
    n_intervals, existing_store, existing_task_state
):  # noqa: D401
    del n_intervals

    now = dt.datetime.now(US_EASTERN)
    today = now.date()
    today_iso = today.isoformat()
    trading_day = _is_trading_day(today)

    store = existing_store if isinstance(existing_store, dict) else {}

    tasks_updates: list[dict[str, T.Any]] = []
    existing_task_map: dict[str, dict[str, T.Any]] = {}
    if isinstance(existing_task_state, dict):
        tasks_payload = existing_task_state.get("tasks")
        if isinstance(tasks_payload, list):
            for task in tasks_payload:
                if isinstance(task, dict) and task.get("id"):
                    existing_task_map[str(task["id"])] = task

    def _task_name(task_id: str) -> str:
        template = TASK_TEMPLATES.get(task_id)
        if isinstance(template, dict) and template.get("name"):
            return str(template["name"])
        return task_id

    def _update_task(task_id: str, status: str, detail: str, **extra: T.Any) -> None:
        payload = {
            "id": task_id,
            "name": _task_name(task_id),
            "status": status,
            "detail": detail,
        }
        payload.update(extra)
        tasks_updates.append(payload)

    def _ensure_prediction_task(task_id: str, detail: str, *, default_status: str = "等待") -> None:
        existing = existing_task_map.get(task_id)
        if existing is not None:
            current_status = str(existing.get("status") or "")
            if current_status not in {"等待", ""}:
                return
            current_detail = str(existing.get("detail") or "")
            if current_detail == detail:
                return
            status_to_use = default_status if current_status in {"", "等待"} else current_status
        else:
            status_to_use = default_status
        _update_task(task_id, status_to_use, detail)
        existing_task_map[task_id] = {"status": status_to_use, "detail": detail}

    def _finalise_tasks():
        if not tasks_updates:
            return no_update
        base_state = existing_task_state if isinstance(existing_task_state, dict) else None
        return _merge_task_updates(base_state, tasks_updates)

    time_now = now.time()
    for key in PREDICTION_TASK_SEQUENCE:
        task_id = f"predict::{key}"
        if key == "decision_day":
            if not trading_day:
                detail = "今日非开盘日，等待下个开盘日 15:00 开始 T+0 预测"
            elif time_now < dt.time(15, 0):
                detail = "等待美东15:00 开始 T+0 预测"
            else:
                detail = "已过美东15:00，可立即开始 T+0 预测"
        elif key == "plus1":
            detail = "待 T+0 预测完成后执行 T+1 预测"
            if not trading_day:
                detail = "待下个开盘日完成 T+0 后执行 T+1 预测"
        elif key == "plus3":
            detail = "待 T+1 预测完成后执行 T+3 预测"
        elif key == "plus7":
            detail = "待 T+3 预测完成后执行 T+7 预测"
        elif key == "plus14":
            detail = "待 T+7 预测完成后执行 T+14 预测"
        else:
            detail = "等待前序预测任务完成"
        _ensure_prediction_task(task_id, detail)

    prev_trading = previous_trading_day(today)
    waiting_detail = (
        f"等待美东10:00 获取 {prev_trading.isoformat()} 的 T+1 行情数据"
    )

    if now.hour < 10:
        _update_task(DAILY_T_MINUS_ONE_TASK_ID, "等待", waiting_detail)
        _update_task(
            DAILY_ADJUST_TASK_ID,
            "等待",
            "等待检验完成后根据结果调整模型参数与因子权重",
        )
        if trading_day:
            backfill_status = "等待"
            backfill_detail = "等待当日预测完成后执行回溯验证"
        else:
            backfill_status = "进行中"
            backfill_detail = "今日非开盘日，持续执行回溯验证"
        _update_task(BACKFILL_TASK_ID, backfill_status, backfill_detail)
        return no_update, _finalise_tasks()

    if store.get("last_run") == today_iso:
        summary = store.get("summary") if isinstance(store.get("summary"), dict) else {}
        target_text = str(
            store.get("date")
            or summary.get("date")
            or prev_trading.isoformat()
        )
        total = int(summary.get("total") or 0)
        correct = int(summary.get("correct") or 0)
        success_rate = summary.get("success_rate")
        try:
            success_pct = f"{round(float(success_rate) * 100, 1)}%"
        except (TypeError, ValueError):
            success_pct = "-"
        errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []

        if total > 0:
            eval_detail = (
                f"{target_text}：命中 {correct}/{total}（{success_pct}）"
            )
            if errors:
                eval_detail += f"，{len(errors)} 条异常"
            eval_status = "已完成"
            adjust_status = "已完成"
            adjust_detail = "已根据最新检验自动调整模型参数与因子权重"
        else:
            message = str(
                store.get("message")
                or summary.get("message")
                or "未找到可用于验证的预测记录。"
            )
            eval_detail = message
            eval_status = "无数据"
            adjust_status = "无数据"
            adjust_detail = "无检验结果，跳过参数调整"

        _update_task(DAILY_T_MINUS_ONE_TASK_ID, eval_status, eval_detail)
        _update_task(DAILY_ADJUST_TASK_ID, adjust_status, adjust_detail)

        backfill_flag = bool(summary.get("backfill") or store.get("backfill"))
        if backfill_flag:
            backfill_status = "进行中"
            backfill_detail = f"回溯验证：最近检验 {target_text}"
        elif not trading_day:
            backfill_status = "进行中"
            backfill_detail = "今日非开盘日，持续执行回溯验证"
        else:
            backfill_status = "等待"
            backfill_detail = "等待当日预测完成后执行回溯验证"
        _update_task(BACKFILL_TASK_ID, backfill_status, backfill_detail)

        return no_update, _finalise_tasks()

    search_start = previous_trading_day(today)
    target_date, is_backfill = _find_validation_target(search_start)
    if target_date is None:
        message = "未找到可用于验证的预测记录。"
        evaluation_store = {
            "last_run": today_iso,
            "date": None,
            "checked_at": now.isoformat(),
            "items": [],
            "summary": {
                "date": None,
                "checked_at": now.isoformat(),
                "total": 0,
                "correct": 0,
                "success_rate": 0.0,
                "avg_move": None,
            },
            "message": message,
        }
        _update_task(DAILY_T_MINUS_ONE_TASK_ID, "无数据", message)
        _update_task(
            DAILY_ADJUST_TASK_ID,
            "无数据",
            "无检验结果，跳过参数调整",
        )
        if trading_day:
            backfill_status = "等待"
            backfill_detail = "等待当日预测完成后执行回溯验证"
        else:
            backfill_status = "进行中"
            backfill_detail = "今日非开盘日，持续执行回溯验证"
        _update_task(BACKFILL_TASK_ID, backfill_status, backfill_detail)
        return evaluation_store, _finalise_tasks()

    archive_entry = _get_prediction_archive(target_date)

    evaluation_items: list[dict[str, T.Any]] = []
    errors: list[str] = []
    correct_count = 0
    move_samples: list[float] = []

    tv_ready = tv_data.is_available()
    tv_error: str | None = None
    if tv_ready:
        try:
            tv_data.ensure_client()
        except tv_data.TVDataError as exc:
            tv_ready = False
            tv_error = str(exc)

    row_lookup: dict[str, dict[str, T.Any]] = {}
    if isinstance(archive_entry, dict):
        rows_payload = archive_entry.get("rowData") or archive_entry.get("rows")
        if isinstance(rows_payload, list):
            for row in rows_payload:
                if not isinstance(row, dict):
                    continue
                symbol_key = str(row.get("symbol") or "").upper()
                if symbol_key and symbol_key not in row_lookup:
                    row_lookup[symbol_key] = row

    results = []
    if isinstance(archive_entry, dict):
        raw_results = archive_entry.get("results")
        if isinstance(raw_results, list):
            results = [
                r
                for r in raw_results
                if isinstance(r, dict)
                and (
                    str(r.get("timeline_key")) == "decision_day"
                    or int(r.get("lookback_days", 0)) == 0
                )
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
        note_detail = f"{target_date.isoformat()}：无预测记录，自动跳过"
        _update_task(DAILY_T_MINUS_ONE_TASK_ID, "无数据", note_detail)
        _update_task(
            DAILY_ADJUST_TASK_ID,
            "无数据",
            "无检验结果，跳过参数调整",
        )
        if is_backfill or not trading_day:
            backfill_status = "进行中"
            backfill_detail = (
                f"回溯验证：待处理 {target_date.isoformat()} 的预测记录"
                if is_backfill
                else "今日非开盘日，持续执行回溯验证"
            )
        else:
            backfill_status = "等待"
            backfill_detail = "等待当日预测完成后执行回溯验证"
        _update_task(BACKFILL_TASK_ID, backfill_status, backfill_detail)
        return evaluation_store, _finalise_tasks()

    for entry in results:
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        row_meta = row_lookup.get(symbol)

        try:
            dci_final = float(entry.get("dci_final") or 0.0)
        except (TypeError, ValueError):
            dci_final = 0.0
        try:
            p_up_pct = float(entry.get("p_up_pct") or 0.0)
        except (TypeError, ValueError):
            p_up_pct = 0.0

        predicted_label = "放弃"
        if dci_final >= 60.0:
            predicted_label = "多" if p_up_pct > 50.0 else "空"

        entry_time, exit_time = _compute_trade_window(target_date)
        actual_reference_date = exit_time.date().isoformat()

        actual_direction = "数据缺失"
        actual_move_pct: float | None = None
        option_move_pct: float | None = None
        correct_text = "-"
        actual_up: bool | None = None
        if predicted_label == "放弃":
            evaluation_source = "skipped"
        else:
            evaluation_source = "tvDatafeed" if tv_ready else "unavailable"

        underlying_entry_price: float | None = None
        underlying_exit_price: float | None = None
        underlying_entry_time: str | None = None
        underlying_exit_time: str | None = None
        option_symbol: str | None = None
        option_exchange: str | None = None
        option_strike: float | None = None
        option_expiry: str | None = None
        option_entry_price: float | None = None
        option_exit_price: float | None = None
        option_entry_time: str | None = None
        option_exit_time: str | None = None

        trade_errors: list[str] = []
        trade_payload: dict[str, T.Any] | None = None

        if predicted_label == "放弃":
            actual_direction = "放弃"
        elif tv_ready:
            trade_payload, trade_errors = _evaluate_trade_window(
                symbol, target_date, predicted_label, row_meta
            )
        elif not tv_ready and tv_error:
            trade_errors.append(tv_error)
        elif not tv_ready:
            trade_errors.append("tvDatafeed 未启用")

        if trade_payload:
            entry_point = trade_payload.get("underlying_entry")
            exit_point = trade_payload.get("underlying_exit")
            evaluation_source = "tvDatafeed"

            if isinstance(entry_point, tv_data.PricePoint):
                underlying_entry_price = entry_point.price
                underlying_entry_time = (
                    entry_point.timestamp.astimezone(US_EASTERN).isoformat()
                )
            if isinstance(exit_point, tv_data.PricePoint):
                underlying_exit_price = exit_point.price
                underlying_exit_time = (
                    exit_point.timestamp.astimezone(US_EASTERN).isoformat()
                )

            option_payload = trade_payload.get("option")
            if isinstance(option_payload, dict):
                option_symbol = option_payload.get("symbol")
                option_exchange = option_payload.get("exchange")
                option_strike = option_payload.get("strike")
                expiry_value = option_payload.get("expiry")
                if isinstance(expiry_value, dt.date):
                    option_expiry = expiry_value.isoformat()
                entry_option = option_payload.get("entry")
                exit_option = option_payload.get("exit")
                if isinstance(entry_option, tv_data.PricePoint):
                    option_entry_price = entry_option.price
                    option_entry_time = (
                        entry_option.timestamp.astimezone(US_EASTERN).isoformat()
                    )
                if isinstance(exit_option, tv_data.PricePoint):
                    option_exit_price = exit_option.price
                    option_exit_time = (
                        exit_option.timestamp.astimezone(US_EASTERN).isoformat()
                    )

                if (
                    isinstance(entry_point, tv_data.PricePoint)
                    and isinstance(exit_point, tv_data.PricePoint)
                    and isinstance(entry_option, tv_data.PricePoint)
                    and isinstance(exit_option, tv_data.PricePoint)
                ):
                    delta_s = exit_point.price - entry_point.price
                    if entry_point.price > 0:
                        actual_move_pct = round(
                            (delta_s / entry_point.price) * 100.0,
                            2,
                        )
                    delta_option = exit_option.price - entry_option.price
                    if entry_option.price > 0:
                        option_move_pct = round(
                            (delta_option / entry_option.price) * 100.0,
                            2,
                        )

                    if predicted_label == "多":
                        if delta_s > 0 and delta_option > 0:
                            actual_direction = "涨"
                            actual_up = True
                        elif delta_s < 0 and delta_option < 0:
                            actual_direction = "跌"
                            actual_up = False
                        else:
                            actual_direction = "放弃"
                            actual_up = None
                    else:  # predicted_label == "空"
                        if delta_s < 0 and delta_option > 0:
                            actual_direction = "涨"
                            actual_up = False
                        elif delta_s > 0 and delta_option < 0:
                            actual_direction = "跌"
                            actual_up = True
                        else:
                            actual_direction = "放弃"
                            actual_up = None

            elif predicted_label != "放弃":
                entry_point = trade_payload.get("underlying_entry")
                exit_point = trade_payload.get("underlying_exit")
                if isinstance(entry_point, tv_data.PricePoint) and isinstance(
                    exit_point, tv_data.PricePoint
                ):
                    delta_s = exit_point.price - entry_point.price
                    if entry_point.price > 0:
                        actual_move_pct = round(
                            (delta_s / entry_point.price) * 100.0,
                            2,
                        )
                    actual_direction = "数据缺失"
                    actual_up = True if delta_s > 0 else False if delta_s < 0 else None

        for err in trade_errors:
            if err:
                errors.append(f"{symbol}: {err}")

        evaluated = predicted_label != "放弃" and actual_direction in {"涨", "跌"}
        if evaluated:
            if actual_direction == "涨":
                correct_text = "是"
                correct_count += 1
            else:
                correct_text = "否"
            if actual_move_pct is not None:
                move_samples.append(abs(actual_move_pct))
        else:
            correct_text = "-"

        item = {
            "symbol": symbol,
            "company": entry.get("company"),
            "sector": entry.get("sector"),
            "predicted_direction": predicted_label,
            "actual_direction": actual_direction,
            "actual_move_pct": actual_move_pct,
            "actual_option_move_pct": option_move_pct,
            "prediction_correct": correct_text,
            "timeline_label": entry.get("timeline_label"),
            "timeline_key": entry.get("timeline_key"),
            "checked_at": now.isoformat(),
            "actual_reference_date": actual_reference_date,
            "evaluation_source": evaluation_source,
            "underlying_entry_price": underlying_entry_price,
            "underlying_exit_price": underlying_exit_price,
            "underlying_entry_time": underlying_entry_time,
            "underlying_exit_time": underlying_exit_time,
            "option_symbol": option_symbol,
            "option_exchange": option_exchange,
            "option_strike": option_strike,
            "option_expiry": option_expiry,
            "option_entry_price": option_entry_price,
            "option_exit_price": option_exit_price,
            "option_entry_time": option_entry_time,
            "option_exit_time": option_exit_time,
        }
        evaluation_items.append(item)

        if (
            RL_MANAGER is not None
            and evaluated
            and actual_up is not None
            and actual_move_pct is not None
        ):
            try:
                RL_MANAGER.apply_feedback(
                    symbol,
                    actual_up=actual_up,
                    actual_move_pct=actual_move_pct,
                    prediction_id=entry.get("rl_prediction_id"),
                    sector=entry.get("sector")
                    if entry.get("sector") not in (None, "未知")
                    else None,
                )
            except Exception:
                pass

    total = sum(
        1
        for item in evaluation_items
        if item.get("predicted_direction") != "放弃"
        and item.get("actual_direction") in {"涨", "跌"}
    )
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
    if is_backfill:
        summary["backfill"] = True

    evaluation_payload = {
        "date": target_date.isoformat(),
        "checked_at": now.isoformat(),
        "items": evaluation_items,
        "summary": summary,
    }
    if is_backfill:
        evaluation_payload["backfill"] = True
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
        evaluation_store["errors"] = errors

    message_lines: list[str] = []
    if total:
        rate_pct = round(success_rate * 100.0, 1)
        message_lines.append(
            f"完成 {target_date.strftime('%Y-%m-%d')} 的决策日预测检验，命中 {correct_count}/{total}（{rate_pct}%）。"
        )
    else:
        message_lines.append("昨日未记录可检验的预测。")
    if errors:
        message_lines.append("；".join(errors))
    evaluation_store["message"] = "\n".join(message_lines)

    if is_backfill:
        note = f"回溯验证：使用 {target_date.isoformat()} 的预测记录"
        if evaluation_store.get("message"):
            evaluation_store["message"] = f"{evaluation_store['message']}; {note}"
        else:
            evaluation_store["message"] = note

    if total:
        eval_detail = (
            f"{target_date.isoformat()}：命中 {correct_count}/{total}"
            f"（{round(success_rate * 100.0, 1)}%）"
        )
        if errors:
            eval_detail += f"，{len(errors)} 条异常"
        eval_status = "已完成"
        adjust_status = "已完成"
        adjust_detail = "已根据最新检验自动调整模型参数与因子权重"
    else:
        eval_detail = f"{target_date.isoformat()}：无预测记录，自动跳过"
        eval_status = "无数据"
        adjust_status = "无数据"
        adjust_detail = "无检验结果，跳过参数调整"

    _update_task(
        DAILY_T_MINUS_ONE_TASK_ID,
        eval_status,
        eval_detail,
        start_time=now.strftime("%H:%M:%S"),
        end_time=now.strftime("%H:%M:%S"),
    )
    _update_task(
        DAILY_ADJUST_TASK_ID,
        adjust_status,
        adjust_detail,
    )

    if is_backfill:
        backfill_status = "进行中"
        backfill_detail = f"回溯验证：处理 {target_date.isoformat()} 的预测记录"
    elif not trading_day:
        backfill_status = "进行中"
        backfill_detail = "今日非开盘日，持续执行回溯验证"
    else:
        backfill_status = "等待"
        backfill_detail = "等待当日预测完成后执行回溯验证"
    _update_task(BACKFILL_TASK_ID, backfill_status, backfill_detail)

    return evaluation_store, _finalise_tasks()


def sync_actual_into_predictions_logic(evaluation_store, prediction_store):  # noqa: D401
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

def render_task_table_logic(task_state):  # noqa: D401
    if not isinstance(task_state, dict):
        return []

    tasks = task_state.get("tasks")
    if not isinstance(tasks, list):
        return []

    return tasks

def render_model_parameters_logic(agent_data):  # noqa: D401
    if not isinstance(agent_data, dict):
        return []

    sectors = agent_data.get("sectors") if isinstance(agent_data.get("sectors"), dict) else {}

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

    return sector_rows

def render_model_details_logic(agent_data):  # noqa: D401
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
    """Normalise RL parameter values for stable change detection."""

    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if math.isnan(number) or math.isinf(number):
            return None
        if isinstance(value, int):
            return int(number)
        return round(number, 6)

    if isinstance(value, (list, tuple, set)):
        return [_normalise_history_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _normalise_history_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    return _json_safe(value)


def _extract_global_trackable(agent_data: dict[str, T.Any]) -> dict[str, T.Any]:
    """Extract comparable global RL parameters from the agent snapshot."""

    if not isinstance(agent_data, dict):
        return {}

    global_data = agent_data.get("global")
    if not isinstance(global_data, dict):
        global_data = agent_data if isinstance(agent_data, dict) else {}

    trackable: dict[str, T.Any] = {}
    scalar_keys = (
        "learning_rate",
        "gamma",
        "adjustment_scale",
        "bias",
        "baseline",
        "update_count",
    )
    for key in scalar_keys:
        if key in global_data:
            trackable[key] = global_data.get(key)

    if "total_predictions" in global_data:
        trackable["total_predictions"] = global_data.get("total_predictions")
    elif "prediction_count" in global_data:
        trackable["total_predictions"] = global_data.get("prediction_count")

    weights = global_data.get("weights")
    if isinstance(weights, dict):
        for feature, weight in sorted(weights.items(), key=lambda item: str(item[0])):
            trackable[f"weight::{feature}"] = weight

    factor_weights = agent_data.get("factor_weights")
    if isinstance(factor_weights, dict):
        for factor, weight in sorted(factor_weights.items(), key=lambda item: str(item[0])):
            trackable[f"factor::{factor}"] = weight

    return trackable


def _extract_sector_trackable(agent_data: dict[str, T.Any]) -> dict[str, dict[str, T.Any]]:
    """Extract comparable sector-specific RL parameters from the agent snapshot."""

    sectors_payload = agent_data.get("sectors") if isinstance(agent_data, dict) else None
    if not isinstance(sectors_payload, dict):
        return {}

    result: dict[str, dict[str, T.Any]] = {}
    for sector_name, payload in sectors_payload.items():
        if not isinstance(payload, dict):
            continue

        sector_entry: dict[str, T.Any] = {}
        for key in ("baseline", "update_count", "total_predictions", "prediction_count"):
            if key in payload:
                sector_entry[key] = payload.get(key)

        weights = payload.get("weights")
        if isinstance(weights, dict):
            for feature, weight in sorted(weights.items(), key=lambda item: str(item[0])):
                sector_entry[f"weight::{feature}"] = weight

        if sector_entry:
            result[str(sector_name)] = sector_entry

    return result


def update_parameter_history_logic(agent_data, history_state, log_state):  # noqa: D401
    base_history: dict[str, T.Any]
    if isinstance(history_state, dict):
        base_history = history_state
    else:
        base_history = {"global": {"last": {}, "changes": []}, "sectors": {}}

    initial_logs = log_state if isinstance(log_state, list) else []
    log_entries = initial_logs

    def _emit(message: str) -> None:
        nonlocal log_entries
        if not message:
            return
        log_entries = append_log(log_entries, message, task_label="强化模型")

    if not isinstance(agent_data, dict):
        return base_history, no_update

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
        summary_bits: list[str] = []
        if recorded_global_changes:
            summary_bits.append(f"全局参数变更 {len(recorded_global_changes)} 项")
        if recorded_sector_changes:
            summary_bits.append(f"行业参数变更 {len(recorded_sector_changes)} 项")
        if summary_bits:
            _emit("RL 参数更新：" + "；".join(summary_bits))

    log_output = log_entries if log_entries != initial_logs else no_update

    return history, log_output

def render_parameter_history_tables_logic(history_state):  # noqa: D401
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

def update_log_output_logic(log_state):
    if not isinstance(log_state, list) or not log_state:
        return "暂无日志记录。"
    return "\n".join(log_state)

def render_validation_view_logic(selected_date, selected_symbol, evaluation_store, prediction_store):  # noqa: D401
    del prediction_store

    archive = _load_prediction_archive_raw()
    overview_total_fig, overview_sector_fig, overview_trend_fig = build_overview_figures(archive)
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
            overview_total_fig,
            overview_sector_fig,
            overview_trend_fig,
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
            overview_total_fig,
            overview_sector_fig,
            overview_trend_fig,
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
        overview_total_fig,
        overview_sector_fig,
        overview_trend_fig,
    )


def _empty_overview_figure(message: str) -> go.Figure:
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


def build_overview_figures(archive: dict[str, T.Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    """根据预测归档构建整体命中率、行业命中率和时间序列趋势图。"""

    if not isinstance(archive, dict) or not archive:
        msg = "暂无历史检验数据。"
        empty_fig = _empty_overview_figure(msg)
        return (empty_fig, empty_fig, empty_fig)

    item_rows: list[dict[str, T.Any]] = []
    summary_rows: list[dict[str, T.Any]] = []

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
        empty_fig = _empty_overview_figure(msg)
        return (empty_fig, empty_fig, empty_fig)

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


def render_overview_charts_logic(prediction_store, evaluation_store):  # noqa: D401
    del prediction_store, evaluation_store
    archive = _load_prediction_archive_raw()
    return build_overview_figures(archive)

def toggle_log_modal_logic(show_clicks, close_clicks, is_open):
    trigger = ctx.triggered_id
    if trigger == "show-log-btn":
        return True
    if trigger == "close-log-btn":
        return False
    return is_open

def download_csv_logic(n, data):
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

