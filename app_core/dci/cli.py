"""Command line helpers for inspecting DCI results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from . import build_inputs, compute_dci
from .providers import DCIDataProviderError, load_dci_payloads


def _load_payload_from_path(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - I/O defensive
        raise SystemExit(f"无法读取输入文件: {exc}")

    if isinstance(data, dict):
        # allow either {symbol: payload} or a single payload with 'factors'
        if any(isinstance(value, dict) for value in data.values()):
            return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        payload: Dict[str, Dict[str, Any]] = {}
        for entry in data:
            if isinstance(entry, dict):
                symbol = str(entry.get("symbol") or entry.get("ticker") or "").upper()
                if symbol:
                    payload[symbol] = entry
        return payload

    if isinstance(data, dict) and "factors" in data:
        return {"": data}  # symbol resolved later

    raise SystemExit("输入文件格式不支持")


def _resolve_payload(symbol: str, path: Path | None) -> Dict[str, Any]:
    if path is not None:
        payload_map = _load_payload_from_path(path)
    else:
        try:
            payload_map = load_dci_payloads()
        except DCIDataProviderError as exc:  # pragma: no cover - provider errors
            raise SystemExit(str(exc))

    if symbol in payload_map:
        return payload_map[symbol]
    if "" in payload_map:
        return payload_map[""]
    available = ", ".join(sorted(payload_map)) or "无"
    raise SystemExit(f"未找到标的 {symbol} 的 DCI 输入，已载入: {available}")


def _format_result(result) -> Dict[str, Any]:
    output = {
        "symbol": result.symbol,
        "direction": result.direction,
        "p_up": result.p_up,
        "dci_base": result.dci_base,
        "dci_penalised": result.dci_penalised,
        "dci_final": result.dci_final,
        "position_weight": result.position_weight,
        "position_bucket": result.position_bucket,
        "certainty": result.certainty,
        "gating_passed": result.gating_passed,
        "gating_reasons": list(result.gating_reasons),
    }
    output["shrink_factors"] = result.shrink_factors
    output["base_score"] = result.base_score
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="计算单个标的的 DCI 分数")
    parser.add_argument("symbol", help="标的代码，例如 AAPL")
    parser.add_argument(
        "--payload",
        type=Path,
        help="指定 JSON 输入文件（默认使用配置提供器）",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="输出完整因子缩放、权重与准入信息",
    )
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    payload = _resolve_payload(symbol, args.payload)
    inputs = build_inputs(symbol, payload)
    result = compute_dci(inputs)

    output = _format_result(result)
    if args.explain:
        output["scaled_factors"] = result.scaled_factors
        output["factor_weights"] = result.factor_weights
        output["shrink_factors"] = result.shrink_factors

    print(json.dumps(output, ensure_ascii=False, indent=2 if args.explain else None))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
