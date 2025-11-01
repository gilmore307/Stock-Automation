"""Reinforcement learning agent that calibrates DCI predictions with feedback."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import threading

from . import (
    BASE_FACTOR_WEIGHTS,
    DCIResult,
    adjust_factor_weights,
    get_factor_weights,
    set_factor_weights,
)

DEFAULT_STATE_FILE = Path(__file__).with_name("dci_rl_state.json")
LEGACY_STATE_PATH = Path(os.getenv("DCI_RL_STATE_PATH", str(DEFAULT_STATE_FILE)))
DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "archives" / "rl_models"
STATE_DIR = Path(os.getenv("DCI_RL_STATE_DIR", str(DEFAULT_STATE_DIR)))
HISTORY_LIMIT = 200


@dataclass(frozen=True)
class RLPrediction:
    """Snapshot of an agent prediction for later feedback."""

    prediction_id: str
    symbol: str
    base_probability: float
    adjusted_probability: float
    direction: int
    adjustment: float


@dataclass
class PendingPrediction:
    """Internal container for predictions awaiting feedback."""

    prediction_id: str
    symbol: str
    timestamp: float
    features: Dict[str, float]
    base_probability: float
    adjusted_probability: float
    direction: int


@dataclass(frozen=True)
class RLFeedbackResult:
    """Summary returned after applying feedback."""

    symbol: str
    prediction_ids: List[str]
    reward: float
    correct: bool
    weight_updates: Dict[str, float]
    baseline: float
    sector: Optional[str] = None
    sector_reward: Optional[float] = None
    sector_correct: Optional[bool] = None
    sector_weight_updates: Optional[Dict[str, float]] = None
    sector_baseline: Optional[float] = None


class DCIRLAgent:
    """A lightweight policy-gradient agent built on top of DCI outputs."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        gamma: float = 0.9,
        adjustment_scale: float = 0.15,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.adjustment_scale = adjustment_scale
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0
        self.baseline: float = 0.0
        self.pending: Dict[str, List[PendingPrediction]] = {}
        self.update_count: int = 0
        self.prediction_count: int = 0

    # ------------------------------------------------------------------
    # Feature extraction and prediction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(result: DCIResult) -> Dict[str, float]:
        features: Dict[str, float] = {
            "bias": 1.0,
            "base_score": float(result.base_score),
            "p_up_offset": float(result.p_up - 0.5),
            "dci_final": float(result.dci_final / 100.0),
            "dci_penalised": float(result.dci_penalised / 100.0),
            "position_weight": float(result.position_weight),
            "certainty": float(result.certainty / 100.0),
        }

        for shrink_key, shrink_value in result.shrink_factors.items():
            features[f"shrink_{shrink_key}"] = float(shrink_value)

        for factor_name, factor_value in result.scaled_factors.items():
            features[f"factor_{factor_name}"] = float(factor_value)

        return features

    def _ensure_feature_keys(self, features: Dict[str, float]) -> None:
        for key in features:
            self.weights.setdefault(key, 0.0)

    def record_prediction(self, result: DCIResult) -> RLPrediction:
        """Store a prediction to await feedback and return adjusted probability."""

        features = self._extract_features(result)
        self._ensure_feature_keys(features)

        activation = self.bias
        for key, value in features.items():
            activation += self.weights.get(key, 0.0) * value

        adjustment = math.tanh(activation) * self.adjustment_scale
        adjusted_probability = max(0.0, min(1.0, result.p_up + adjustment))
        direction = 1 if adjusted_probability >= 0.5 else -1

        prediction = PendingPrediction(
            prediction_id=uuid.uuid4().hex,
            symbol=result.symbol,
            timestamp=time.time(),
            features=features,
            base_probability=result.p_up,
            adjusted_probability=adjusted_probability,
            direction=direction,
        )
        self.pending.setdefault(result.symbol, []).append(prediction)
        self.prediction_count += 1

        return RLPrediction(
            prediction_id=prediction.prediction_id,
            symbol=result.symbol,
            base_probability=result.p_up,
            adjusted_probability=adjusted_probability,
            direction=direction,
            adjustment=adjusted_probability - result.p_up,
        )

    # ------------------------------------------------------------------
    # Feedback and training
    # ------------------------------------------------------------------
    def apply_feedback(
        self,
        symbol: str,
        actual_up: bool,
        actual_move_pct: Optional[float] = None,
        prediction_id: Optional[str] = None,
    ) -> RLFeedbackResult:
        """Update the agent using actual outcomes."""

        queue = self.pending.get(symbol.upper()) or self.pending.get(symbol)
        if not queue:
            raise ValueError(f"No pending prediction for symbol {symbol}.")

        if prediction_id:
            targets = [p for p in queue if p.prediction_id == prediction_id]
            if not targets:
                raise ValueError(
                    f"No pending prediction {prediction_id} for symbol {symbol}."
                )
        else:
            targets = list(queue)

        target_value = 1.0 if actual_up else 0.0

        reward_magnitude = 1.0
        if actual_move_pct is not None:
            reward_magnitude = math.tanh(abs(actual_move_pct) / 10.0) or 1.0

        correct = False
        weight_updates: Dict[str, float] = {}
        reward_total = 0.0

        for pending_prediction in targets:
            correct = (
                correct
                or (actual_up and pending_prediction.direction > 0)
                or ((not actual_up) and pending_prediction.direction < 0)
            )
            reward = reward_magnitude
            if actual_up != (pending_prediction.direction > 0):
                reward = -reward
            reward_total += reward

            gradient = target_value - pending_prediction.adjusted_probability
            advantage = reward - self.baseline
            update_factor = self.learning_rate * (gradient + advantage)

            for key, value in pending_prediction.features.items():
                old_weight = self.weights.get(key, 0.0)
                new_weight = old_weight + update_factor * value
                self.weights[key] = new_weight
                weight_updates[key] = new_weight - old_weight

            self.bias += update_factor
            self.baseline = self.gamma * self.baseline + (1.0 - self.gamma) * reward
            self.update_count += 1

        # remove processed predictions
        if prediction_id:
            self.pending[symbol] = [p for p in queue if p.prediction_id != prediction_id]
            if not self.pending[symbol]:
                self.pending.pop(symbol, None)
        else:
            self.pending.pop(symbol, None)

        avg_reward = reward_total / len(targets)

        return RLFeedbackResult(
            symbol=symbol.upper(),
            prediction_ids=[p.prediction_id for p in targets],
            reward=avg_reward,
            correct=correct,
            weight_updates=weight_updates,
            baseline=self.baseline,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serialisable snapshot of the agent state."""

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "adjustment_scale": self.adjustment_scale,
            "bias": self.bias,
            "baseline": self.baseline,
            "update_count": self.update_count,
            "weights": dict(self.weights),
            "pending_counts": {symbol: len(preds) for symbol, preds in self.pending.items()},
            "total_predictions": self.prediction_count,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "adjustment_scale": self.adjustment_scale,
            "bias": self.bias,
            "baseline": self.baseline,
            "update_count": self.update_count,
            "prediction_count": self.prediction_count,
            "weights": self.weights,
            "pending": [
                {
                    "prediction_id": p.prediction_id,
                    "symbol": p.symbol,
                    "timestamp": p.timestamp,
                    "features": p.features,
                    "base_probability": p.base_probability,
                    "adjusted_probability": p.adjusted_probability,
                    "direction": p.direction,
                }
                for preds in self.pending.values()
                for p in preds
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DCIRLAgent":
        agent = cls(
            learning_rate=float(payload.get("learning_rate", 0.05)),
            gamma=float(payload.get("gamma", 0.9)),
            adjustment_scale=float(payload.get("adjustment_scale", 0.15)),
        )
        agent.bias = float(payload.get("bias", 0.0))
        agent.baseline = float(payload.get("baseline", 0.0))
        agent.update_count = int(payload.get("update_count", 0))
        agent.prediction_count = int(payload.get("prediction_count", 0))

        weights = payload.get("weights")
        if isinstance(weights, dict):
            agent.weights = {str(k): float(v) for k, v in weights.items()}

        pending_payload = payload.get("pending")
        if isinstance(pending_payload, list):
            for entry in pending_payload:
                if not isinstance(entry, dict):
                    continue
                prediction = PendingPrediction(
                    prediction_id=str(entry.get("prediction_id", uuid.uuid4().hex)),
                    symbol=str(entry.get("symbol", "")).upper(),
                    timestamp=float(entry.get("timestamp", time.time())),
                    features={str(k): float(v) for k, v in (entry.get("features") or {}).items()},
                    base_probability=float(entry.get("base_probability", 0.5)),
                    adjusted_probability=float(entry.get("adjusted_probability", 0.5)),
                    direction=int(entry.get("direction", 1)),
                )
                agent.pending.setdefault(prediction.symbol, []).append(prediction)

        return agent

    # ------------------------------------------------------------------
    # File persistence API
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "DCIRLAgent":
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(payload, dict):
            return cls()
        return cls.from_dict(payload)


class RLAgentManager:
    """Thread-safe singleton-like manager for the RL agent."""

    def __init__(
        self,
        state_path: Path = LEGACY_STATE_PATH,
        state_dir: Path = STATE_DIR,
        factor_weight_rate: float = 0.05,
    ) -> None:
        self.legacy_state_path = Path(state_path)
        self.state_dir = Path(state_dir)
        self._lock = threading.Lock()
        self._agent = DCIRLAgent()
        self._sector_agents: Dict[str, DCIRLAgent] = {}
        self.factor_weight_rate = float(factor_weight_rate)
        self._factor_weights: Dict[str, float] = get_factor_weights()
        self._sector_file_map: Dict[str, Path] = {}
        self._model_history_cache: Dict[str, List[Dict[str, object]]] = {}
        self._last_saved_models: Dict[str, Dict[str, object]] = {}
        self._factor_history_cache: List[Dict[str, object]] = []
        self._last_saved_factor_weights: Dict[str, float] = {}
        self._load_state()
        # Ensure the DCI module sees the persisted weights when the manager starts.
        set_factor_weights(self._factor_weights)
        self._model_history_cache.setdefault("global", [])
        if "global" not in self._last_saved_models:
            self._last_saved_models["global"] = copy.deepcopy(self._agent.to_dict())
        if not self._last_saved_factor_weights:
            self._last_saved_factor_weights = dict(self._factor_weights)

    @staticmethod
    def _current_timestamp() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def _sector_key(sector: str) -> str:
        return f"sector::{sector}"

    @staticmethod
    def _safe_sector_file_name(sector: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in sector.strip())
        safe = safe[:40] or "sector"
        digest = hashlib.sha1(sector.encode("utf-8")).hexdigest()[:8]
        return f"sector_{safe}_{digest}.json"

    @staticmethod
    def _values_equal(old: object, new: object) -> bool:
        if isinstance(old, (int, float)) or isinstance(new, (int, float)):
            try:
                return math.isclose(float(old), float(new), rel_tol=1e-9, abs_tol=1e-12)
            except (TypeError, ValueError):
                return False
        return old == new

    def _get_sector_path(self, sector: str) -> Path:
        if sector in self._sector_file_map:
            return self._sector_file_map[sector]
        filename = self._safe_sector_file_name(sector)
        path = self.state_dir / filename
        self._sector_file_map[sector] = path
        return path

    def _compute_model_changes(
        self,
        previous: Optional[Dict[str, object]],
        current: Dict[str, object],
    ) -> List[Dict[str, object]]:
        if not previous:
            return []

        changes: List[Dict[str, object]] = []
        for key in (
            "learning_rate",
            "gamma",
            "adjustment_scale",
            "bias",
            "baseline",
            "update_count",
            "prediction_count",
        ):
            old_value = previous.get(key) if isinstance(previous, dict) else None
            new_value = current.get(key)
            if not self._values_equal(old_value, new_value):
                changes.append({
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": new_value,
                })

        previous_weights = (
            previous.get("weights") if isinstance(previous, dict) and isinstance(previous.get("weights"), dict) else {}
        )
        current_weights = (
            current.get("weights") if isinstance(current.get("weights"), dict) else {}
        )
        weight_keys = set(previous_weights) | set(current_weights)
        for weight_key in sorted(weight_keys):
            old_weight = previous_weights.get(weight_key)
            new_weight = current_weights.get(weight_key)
            if not self._values_equal(old_weight, new_weight):
                changes.append({
                    "parameter": f"weight::{weight_key}",
                    "old_value": old_weight,
                    "new_value": new_weight,
                })

        return changes

    def _write_model_file(
        self,
        path: Path,
        cache_key: str,
        model_type: str,
        state: Dict[str, object],
        **metadata: object,
    ) -> None:
        timestamp = self._current_timestamp()
        history = self._model_history_cache.get(cache_key, [])
        changes = self._compute_model_changes(self._last_saved_models.get(cache_key), state)
        if changes:
            history = (history + [{"timestamp": timestamp, "changes": changes}])[-HISTORY_LIMIT:]
        payload = {
            "model": model_type,
            "state": state,
            "history": history,
            "updated_at": timestamp,
            **metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        self._model_history_cache[cache_key] = history
        self._last_saved_models[cache_key] = copy.deepcopy(state)

    def _write_factor_weights_file(self) -> None:
        timestamp = self._current_timestamp()
        previous = self._last_saved_factor_weights or {}
        changes: List[Dict[str, object]] = []
        all_keys = set(previous) | set(self._factor_weights)
        for key in sorted(all_keys):
            old_value = previous.get(key)
            new_value = self._factor_weights.get(key)
            if not self._values_equal(old_value, new_value):
                changes.append({
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": new_value,
                })
        history = self._factor_history_cache
        if changes:
            history = (history + [{"timestamp": timestamp, "changes": changes}])[-HISTORY_LIMIT:]
        payload = {
            "updated_at": timestamp,
            "weights": self._factor_weights,
            "history": history,
        }
        factor_path = self.state_dir / "factor_weights.json"
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        with factor_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        self._factor_history_cache = history
        self._last_saved_factor_weights = dict(self._factor_weights)

    def _write_legacy_state_file(self) -> None:
        payload = {
            "global": self._agent.to_dict(),
            "sectors": {sector: agent.to_dict() for sector, agent in self._sector_agents.items()},
            "factor_weights": self._factor_weights,
        }
        self.legacy_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.legacy_state_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)

    def _load_state(self) -> None:
        if self._load_from_directory():
            return
        self._load_legacy_state()

    def _load_from_directory(self) -> bool:
        if not self.state_dir.exists():
            return False

        loaded_any = False
        self._sector_agents.clear()
        self._sector_file_map.clear()

        global_path = self.state_dir / "global.json"
        if global_path.exists():
            try:
                with global_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                state_payload = payload.get("state")
                if isinstance(state_payload, dict):
                    self._agent = DCIRLAgent.from_dict(state_payload)
                    self._last_saved_models["global"] = copy.deepcopy(state_payload)
                    history_payload = payload.get("history")
                    if isinstance(history_payload, list):
                        self._model_history_cache["global"] = history_payload[-HISTORY_LIMIT:]
                    loaded_any = True

        for sector_path in sorted(self.state_dir.glob("sector_*.json")):
            try:
                with sector_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            sector_name = payload.get("name") or payload.get("sector")
            if not isinstance(sector_name, str) or not sector_name.strip():
                continue
            state_payload = payload.get("state")
            if not isinstance(state_payload, dict):
                continue
            sector_name = sector_name.strip()
            self._sector_agents[sector_name] = DCIRLAgent.from_dict(state_payload)
            cache_key = self._sector_key(sector_name)
            self._sector_file_map[sector_name] = sector_path
            self._last_saved_models[cache_key] = copy.deepcopy(state_payload)
            history_payload = payload.get("history")
            if isinstance(history_payload, list):
                self._model_history_cache[cache_key] = history_payload[-HISTORY_LIMIT:]
            loaded_any = True

        factor_path = self.state_dir / "factor_weights.json"
        if factor_path.exists():
            try:
                with factor_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                weights_payload = payload.get("weights")
                if isinstance(weights_payload, dict):
                    try:
                        self._factor_weights = set_factor_weights(weights_payload)
                    except Exception:  # pragma: no cover - defensive fallback
                        self._factor_weights = dict(BASE_FACTOR_WEIGHTS)
                    self._last_saved_factor_weights = dict(self._factor_weights)
                history_payload = payload.get("history")
                if isinstance(history_payload, list):
                    self._factor_history_cache = history_payload[-HISTORY_LIMIT:]
                loaded_any = True

        return loaded_any

    def _load_legacy_state(self) -> None:
        if not self.legacy_state_path.exists():
            return
        try:
            with self.legacy_state_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        factor_weights_payload = payload.get("factor_weights")
        if isinstance(factor_weights_payload, dict):
            try:
                self._factor_weights = set_factor_weights(factor_weights_payload)
            except Exception:  # pragma: no cover - defensive fallback
                self._factor_weights = dict(BASE_FACTOR_WEIGHTS)
        self._last_saved_factor_weights = dict(self._factor_weights)

        self._sector_agents.clear()
        if "global" in payload or "sectors" in payload:
            global_payload = payload.get("global")
            if isinstance(global_payload, dict):
                self._agent = DCIRLAgent.from_dict(global_payload)
                self._last_saved_models["global"] = copy.deepcopy(global_payload)
            sectors_payload = payload.get("sectors")
            if isinstance(sectors_payload, dict):
                for sector_name, data in sectors_payload.items():
                    if isinstance(data, dict):
                        sector_name_str = str(sector_name)
                        self._sector_agents[sector_name_str] = DCIRLAgent.from_dict(data)
                        cache_key = self._sector_key(sector_name_str)
                        self._last_saved_models[cache_key] = copy.deepcopy(data)
        else:
            # Backward compatibility with legacy single-agent state
            self._agent = DCIRLAgent.from_dict(payload)
            self._last_saved_models["global"] = copy.deepcopy(payload)

    def _prune_removed_sector_files(self, active_paths: set[Path]) -> None:
        if not self.state_dir.exists():
            return
        for sector_path in self.state_dir.glob("sector_*.json"):
            if sector_path not in active_paths:
                try:
                    sector_path.unlink()
                except OSError:
                    continue
        # Remove stale mappings
        self._sector_file_map = {
            sector: path for sector, path in self._sector_file_map.items() if path in active_paths
        }
        active_sector_keys = {self._sector_key(sector) for sector in self._sector_agents}
        active_sector_keys.add("global")
        self._last_saved_models = {
            key: value
            for key, value in self._last_saved_models.items()
            if key in active_sector_keys
        }
        self._model_history_cache = {
            key: value
            for key, value in self._model_history_cache.items()
            if key in active_sector_keys
        }

    def _save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        global_state = self._agent.to_dict()
        self._write_model_file(
            self.state_dir / "global.json",
            "global",
            "global",
            global_state,
        )

        active_paths: set[Path] = set()
        for sector, agent in self._sector_agents.items():
            sector_state = agent.to_dict()
            path = self._get_sector_path(sector)
            self._write_model_file(
                path,
                self._sector_key(sector),
                "sector",
                sector_state,
                name=sector,
            )
            active_paths.add(path)

        self._prune_removed_sector_files(active_paths)
        self._write_factor_weights_file()
        self._write_legacy_state_file()

    def _get_sector_agent(self, sector: str) -> DCIRLAgent:
        key = sector.strip()
        agent = self._sector_agents.get(key)
        if agent is None:
            agent = DCIRLAgent()
            self._sector_agents[key] = agent
        return agent

    def record_prediction(self, result: DCIResult, sector: Optional[str] = None) -> RLPrediction:
        with self._lock:
            prediction = self._agent.record_prediction(result)
            if sector:
                agent = self._get_sector_agent(sector)
                agent.record_prediction(result)
            self._save_state()
            return prediction

    def apply_feedback(
        self,
        symbol: str,
        actual_up: bool,
        actual_move_pct: Optional[float] = None,
        prediction_id: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> RLFeedbackResult:
        with self._lock:
            feedback = self._agent.apply_feedback(
                symbol=symbol,
                actual_up=actual_up,
                actual_move_pct=actual_move_pct,
                prediction_id=prediction_id,
            )

            sector_feedback: Optional[RLFeedbackResult] = None
            sector_key: Optional[str] = None
            if sector:
                sector_key = sector.strip()
                agent = self._sector_agents.get(sector_key)
                if agent is not None:
                    try:
                        sector_feedback = agent.apply_feedback(
                            symbol=symbol,
                            actual_up=actual_up,
                            actual_move_pct=actual_move_pct,
                            prediction_id=prediction_id,
                        )
                    except ValueError:
                        sector_feedback = None

            factor_updates: Dict[str, float] = {}
            for feature_name, delta in (feedback.weight_updates or {}).items():
                if feature_name.startswith("factor_"):
                    factor_key = feature_name.split("factor_", 1)[1]
                    factor_updates[factor_key] = factor_updates.get(factor_key, 0.0) + float(delta)

            if factor_updates:
                self._factor_weights = adjust_factor_weights(
                    factor_updates,
                    learning_rate=self.factor_weight_rate,
                )

            self._save_state()

            return RLFeedbackResult(
                symbol=feedback.symbol,
                prediction_ids=feedback.prediction_ids,
                reward=feedback.reward,
                correct=feedback.correct,
                weight_updates=feedback.weight_updates,
                baseline=feedback.baseline,
                sector=sector_key,
                sector_reward=sector_feedback.reward if sector_feedback else None,
                sector_correct=sector_feedback.correct if sector_feedback else None,
                sector_weight_updates=sector_feedback.weight_updates if sector_feedback else None,
                sector_baseline=sector_feedback.baseline if sector_feedback else None,
            )

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "global": self._agent.snapshot(),
                "sectors": {sector: agent.snapshot() for sector, agent in self._sector_agents.items()},
                "factor_weights": dict(self._factor_weights),
            }


_GLOBAL_MANAGER: Optional[RLAgentManager] = None
_MANAGER_LOCK = threading.Lock()


def get_global_manager() -> RLAgentManager:
    global _GLOBAL_MANAGER
    with _MANAGER_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = RLAgentManager()
        return _GLOBAL_MANAGER
