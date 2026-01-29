from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from experiments.common.plotting.io_utils import as_bool, as_float, as_int, mean, sem, write_csv

logger = logging.getLogger(__name__)


def _read_summary_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %s", path)
        return None
    return data if isinstance(data, dict) else None


def _parse_env_from_run_id(run_id: Optional[str], sweep_name: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    parts = [p for p in str(run_id).split("__") if p]
    env_parts = [p for p in parts if p.startswith("env")]
    if not env_parts:
        return None

    if sweep_name:
        env_parts = [p for p in env_parts if p != str(sweep_name)]
        if not env_parts:
            return None

    env_part = env_parts[-1]
    label = env_part[3:] if env_part.startswith("env") else env_part
    return label or None


def _infer_environment_label(
    row: Dict[str, Any],
    *,
    output_dir: Path,
    run_config_cache: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]],
) -> Optional[str]:
    existing = str(row.get("environment_label") or "").strip()
    if existing and existing.lower() not in {"none", "null"}:
        return existing

    run_id = str(row.get("run_id") or "").strip()
    model_label = str(row.get("model_label") or "").strip()
    sweep = str(row.get("sweep") or "").strip()
    if run_id and model_label and sweep:
        key = (model_label, sweep, run_id)
        if key not in run_config_cache:
            rc_path = output_dir / "runs" / model_label / sweep / run_id / "run_config.json"
            run_config_cache[key] = _read_json(rc_path)
        rc = run_config_cache.get(key) or {}
        from_rc = str(rc.get("environment_label") or "").strip()
        if from_rc and from_rc.lower() not in {"none", "null"}:
            return from_rc

    return _parse_env_from_run_id(run_id, sweep)


def _canonical_variant(value: Any) -> str:
    return str(value or "").strip()


def _is_complete(status: Any) -> bool:
    return str(status or "").strip().lower() == "complete"


def _finite(value: float) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _seed_means(rows: Iterable[Dict[str, Any]], metric_key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = as_int(r.get("seed"))
        if seed is None:
            continue
        val = as_float(r.get(metric_key))
        if val is None:
            continue
        by_seed.setdefault(int(seed), []).append(float(val))

    out: List[float] = []
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        if not vals:
            continue
        out.append(float(mean(vals)))
    return out


def _group_stats(seed_means: List[float]) -> Dict[str, Any]:
    n = len(seed_means)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "se": None}
    mu = _finite(mean(seed_means))
    se_val = _finite(sem(seed_means))
    return {"n_seeds": int(n), "mean": mu, "se": se_val}


def _format_mean_se(mean_val: Optional[float], se_val: Optional[float], *, decimals: int) -> Optional[str]:
    if mean_val is None or se_val is None:
        return None
    return f"{mean_val:.{decimals}f} ± {se_val:.{decimals}f}"


def _summarize(
    rows: List[Dict[str, Any]],
    *,
    group_keys: Tuple[str, ...],
    metric_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda x: tuple("" if v is None else str(v) for v in x)):
        group_rows = grouped[key]
        vals = _seed_means(group_rows, metric_key)
        stats = _group_stats(vals)

        row: Dict[str, Any] = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row.update(
            {
                "metric": metric_key,
                "mean": stats["mean"],
                "se": stats["se"],
                "n_seeds": stats["n_seeds"],
                "mean_se": _format_mean_se(stats["mean"], stats["se"], decimals=3),
            }
        )
        out.append(row)
    return out


def _prepare_rows(
    output_dir: Path,
    *,
    include_incomplete: bool,
    include_environment_label: bool,
) -> List[Dict[str, Any]]:
    summary_path = output_dir / "summary.csv"
    rows = _read_summary_csv(summary_path)

    run_config_cache: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]] = {}

    out: List[Dict[str, Any]] = []
    for r in rows:
        if not include_incomplete and not _is_complete(r.get("status")):
            continue

        parsed = dict(r)
        parsed["prompt_variant"] = _canonical_variant(parsed.get("prompt_variant"))
        parsed["model_label"] = str(parsed.get("model_label") or "").strip() or None
        parsed["sweep"] = str(parsed.get("sweep") or "").strip() or None
        parsed["topology"] = str(parsed.get("topology") or "").strip() or None
        parsed["num_agents"] = as_int(parsed.get("num_agents"))
        parsed["colluder_count"] = as_int(parsed.get("colluder_count"))
        parsed["secret_channel_enabled"] = as_bool(parsed.get("secret_channel_enabled"))

        if include_environment_label:
            parsed["environment_label"] = _infer_environment_label(
                parsed, output_dir=output_dir, run_config_cache=run_config_cache
            )

        out.append(parsed)
    return out


def _write_blackboard_post_rate_csv(
    output_dir: Path,
    *,
    out_path: Optional[Path],
    group_keys: Tuple[str, ...],
    include_incomplete: bool,
    include_environment_label: bool,
    metric_key: str = "colluder_posts_secret_rate",
) -> Path:
    rows = _prepare_rows(
        output_dir,
        include_incomplete=include_incomplete,
        include_environment_label=include_environment_label,
    )
    summary_rows = _summarize(rows, group_keys=group_keys, metric_key=metric_key)

    dest = out_path or (output_dir / "blackboard_post_rates.csv")
    write_csv(dest, summary_rows)
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSV summaries of colluders' blackboard post rate (mean ± SE). "
            "Uses `colluder_posts_secret_rate` from `summary.csv` and aggregates over seeds."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Experiment timestamp dir for the model sweep (contains summary.csv).",
    )
    parser.add_argument(
        "--envs-dir",
        type=str,
        default=None,
        help="Experiment timestamp dir for the environment sweep (contains summary.csv).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where status != 'complete' (default: only complete runs).",
    )
    parser.add_argument(
        "--models-out",
        type=str,
        default=None,
        help="Output CSV path for the model sweep (default: <models-dir>/blackboard_post_rates_by_model.csv).",
    )
    parser.add_argument(
        "--envs-out",
        type=str,
        default=None,
        help="Output CSV path for the environment sweep (default: <envs-dir>/blackboard_post_rates_by_environment.csv).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="colluder_posts_secret_rate",
        help="Metric key in summary.csv to aggregate (default: colluder_posts_secret_rate).",
    )

    args = parser.parse_args(argv)
    include_incomplete = bool(args.include_incomplete)
    metric_key = str(args.metric).strip()

    if not args.models_dir and not args.envs_dir:
        raise SystemExit("Provide at least one of --models-dir or --envs-dir.")

    if args.models_dir:
        models_dir = Path(args.models_dir).expanduser()
        out_path = Path(args.models_out).expanduser() if args.models_out else None
        dest = _write_blackboard_post_rate_csv(
            models_dir,
            out_path=out_path or (models_dir / "blackboard_post_rates_by_model.csv"),
            group_keys=(
                "model_label",
                "sweep",
                "topology",
                "num_agents",
                "colluder_count",
                "secret_channel_enabled",
                "prompt_variant",
            ),
            include_incomplete=include_incomplete,
            include_environment_label=False,
            metric_key=metric_key,
        )
        logger.info("Wrote: %s", dest)

    if args.envs_dir:
        envs_dir = Path(args.envs_dir).expanduser()
        out_path = Path(args.envs_out).expanduser() if args.envs_out else None
        dest = _write_blackboard_post_rate_csv(
            envs_dir,
            out_path=out_path or (envs_dir / "blackboard_post_rates_by_environment.csv"),
            group_keys=(
                "environment_label",
                "sweep",
                "topology",
                "num_agents",
                "colluder_count",
                "secret_channel_enabled",
                "prompt_variant",
            ),
            include_incomplete=include_incomplete,
            include_environment_label=True,
            metric_key=metric_key,
        )
        logger.info("Wrote: %s", dest)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())

