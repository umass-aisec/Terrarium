from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SweepLabels:
    experiment_tag: str
    timestamp: str
    model_label: str
    sweep_name: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(path: Path) -> Optional[Any]:
    try:
        return load_json(path)
    except FileNotFoundError:
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_run_dirs(sweep_dir: Path) -> Iterable[Path]:
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "run_config.json").exists():
            yield child


def infer_labels_from_sweep_dir(sweep_dir: Path) -> SweepLabels:
    """
    Best-effort parse:
      experiments/<project>/outputs/<tag>/<timestamp>/runs/<model_label>/<sweep_name>
    """
    parts = list(sweep_dir.parts)
    sweep_name = sweep_dir.name
    model_label = sweep_dir.parent.name if sweep_dir.parent else "unknown_model"

    experiment_tag = "unknown_experiment"
    timestamp = "unknown_timestamp"

    try:
        runs_idx = parts.index("runs")
        if runs_idx >= 2:
            timestamp = parts[runs_idx - 1]
            experiment_tag = parts[runs_idx - 2]
    except ValueError:
        pass

    def _sanitize(value: str) -> str:
        return (
            "".join(
                c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in value
            ).strip("_")
            or "unknown"
        )

    return SweepLabels(
        experiment_tag=_sanitize(experiment_tag),
        timestamp=_sanitize(timestamp),
        model_label=_sanitize(model_label),
        sweep_name=_sanitize(sweep_name),
    )


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def groupby(
    rows: List[Dict[str, Any]], keys: Tuple[str, ...]
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        k = tuple(row.get(col) for col in keys)
        grouped.setdefault(k, []).append(row)
    return grouped


def mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def std(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return float(var**0.5)


def sem(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return std(values) / (len(values) ** 0.5)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}
