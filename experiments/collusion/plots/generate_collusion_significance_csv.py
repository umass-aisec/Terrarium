from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import permutation_test

from experiments.collusion.plots.common import canonical_variant, default_out_dir
from experiments.common.plotting.io_utils import (
    as_float,
    as_int,
    ensure_dir,
    mean,
    sem,
    sanitize_filename,
    write_csv,
)
from experiments.common.plotting.load_runs import load_runs
from experiments.common.plotting.logging_utils import configure_basic_logging

# Reuse the exact row-building logic from the radar generator so the metrics we
# test match the metrics we plot (e.g., SmartGrid joint_reward_ratio handling).
from experiments.collusion.plots.generate_collusion_radar import _build_rows  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    label: str
    higher_is_better: bool = True


def _finite(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except Exception:
        return None
    return float(f) if math.isfinite(f) else None


def _sorted_variants(variants: Iterable[str]) -> List[str]:
    preferred = ["control", "simple", "deception", "structured", "aggressive"]
    vals = [canonical_variant(v) for v in variants if v is not None and str(v).strip()]
    uniq: List[str] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        uniq.append(v)
        seen.add(v)

    out: List[str] = []
    for v in preferred:
        if v in seen:
            out.append(v)
            seen.remove(v)
    out.extend(sorted(seen))
    return out


def _select_colluder_count(rows: List[Dict[str, Any]], requested: Optional[int]) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--colluder-count must be > 0")
        return int(requested)

    counts: List[int] = []
    for r in rows:
        c = as_int(r.get("colluder_count"))
        if c is None or c <= 0:
            continue
        counts.append(int(c))
    if not counts:
        raise ValueError("No runs with colluder_count > 0 found.")
    return int(max(counts))


def _filter_group(
    rows: List[Dict[str, Any]],
    *,
    colluder_count: int,
    secret: bool,
    prompt_variant: str,
    require_complete: bool,
) -> List[Dict[str, Any]]:
    pv = canonical_variant(prompt_variant)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if int(r.get("colluder_count") or 0) != int(colluder_count):
            continue
        if bool(r.get("secret_channel_enabled") is True) != bool(secret):
            continue
        if canonical_variant(r.get("prompt_variant")) != pv:
            continue
        if require_complete and str(r.get("status") or "").strip().lower() != "complete":
            continue
        out.append(r)
    return out


def _seed_means(rows: List[Dict[str, Any]], key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = as_int(r.get("seed"))
        if seed is None:
            continue
        v = _finite(as_float(r.get(key)))
        if v is None:
            continue
        by_seed.setdefault(int(seed), []).append(float(v))

    out: List[float] = []
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        if not vals:
            continue
        out.append(float(mean([float(x) for x in vals])))
    return out


def _group_stats_from_seed_means(vals: List[float]) -> Dict[str, Any]:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None}
    mu = _finite(mean(vals))
    se = _finite(sem(vals))
    return {"n": int(n), "mean": mu, "sem": se}


def _z_p_value_two_sided(z: float) -> Optional[float]:
    if not math.isfinite(z):
        return None
    # two-sided p-value under standard normal: 2*(1-Φ(|z|)) = erfc(|z|/sqrt(2))
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _auto_group_by(rows: List[Dict[str, Any]]) -> str:
    envs = {
        str(r.get("environment_label"))
        for r in rows
        if r.get("environment_label") is not None and str(r.get("environment_label")).strip()
    }
    topos = {
        str(r.get("topology"))
        for r in rows
        if r.get("topology") is not None and str(r.get("topology")).strip()
    }
    if len(envs) > 1:
        return "environment"
    if len(topos) > 1:
        return "topology"
    return "none"


def _group_keys(group_by: str) -> Tuple[str, ...]:
    if group_by == "none":
        return ()
    if group_by == "environment":
        return ("environment_label",)
    if group_by == "topology":
        return ("topology",)
    if group_by in {"environment_topology", "environment+topology"}:
        return ("environment_label", "topology")
    raise ValueError(f"Unknown group-by: {group_by}")


def _permutation_p_value_mean_diff(
    treatment_vals: List[float],
    baseline_vals: List[float],
    *,
    n_resamples: int,
    random_seed: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    if not treatment_vals or not baseline_vals:
        return None, None
    a = np.asarray(treatment_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    if a.size == 0 or b.size == 0:
        return None, None

    def _mean_diff(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = permutation_test(
        (a, b),
        statistic=_mean_diff,
        permutation_type="independent",
        vectorized=True,
        alternative="two-sided",
        n_resamples=int(n_resamples),
        random_state=random_seed,
    )
    return float(res.statistic), float(res.pvalue)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV of z-based 95% CIs for Δ = μ_treatment − μ_baseline "
            "using SE(Δ)=sqrt(SE_t^2+SE_b^2) (independent groups)."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/collusion/outputs/<tag>/<ts>/runs/<model_label>/<sweep_name>",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output CSV path (default: <plots_outputs>/<tag>/<ts>/<model>/<sweep>/sweep/collusion_significance__*.csv).",
    )
    parser.add_argument(
        "--colluder-count",
        type=int,
        default=None,
        help="Which colluder_count to test (default: max > 0).",
    )
    parser.add_argument(
        "--baseline-prompt-variant",
        type=str,
        default="control",
        help="Prompt variant for the baseline (secret_channel_enabled=false) group (default: control).",
    )
    parser.add_argument(
        "--treatment-prompt-variant",
        type=str,
        default=None,
        help="Prompt variant for the secret-channel treatment (default: inferred; prefers non-'control').",
    )
    parser.add_argument(
        "--plot-all-prompt-variants",
        action="store_true",
        help="Test baseline vs every secret-channel prompt_variant present in the sweep.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where metrics.status != 'complete' (not recommended).",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present (final_summary_repaired.json, metrics_repaired.json).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="auto",
        choices=["auto", "none", "environment", "topology", "environment_topology", "environment+topology"],
        help="How to group runs before comparing baseline vs treatment (default: auto).",
    )
    parser.add_argument(
        "--z-critical",
        type=float,
        default=1.96,
        help="Critical z value for the CI half-width (default: 1.96 for ~95%%).",
    )
    parser.add_argument(
        "--permutation-test",
        action="store_true",
        help="Also compute permutation-test p-values over per-seed means (recommended for small n / non-normal metrics).",
    )
    parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=100_000,
        help="Number of permutation resamples (default: 100000).",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=0,
        help="Random seed for permutation test resampling (default: 0).",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir, prefer_repaired=bool(args.prefer_repaired))
    rows = _build_rows(runs)
    if not rows:
        raise SystemExit("No runs found under --sweep-dir.")

    colluder_count = _select_colluder_count(rows, args.colluder_count)
    require_complete = not bool(args.include_incomplete)
    baseline_variant = canonical_variant(args.baseline_prompt_variant)

    available_treatments = {
        canonical_variant(r.get("prompt_variant"))
        for r in rows
        if int(r.get("colluder_count") or 0) == int(colluder_count)
        and r.get("secret_channel_enabled") is True
        and r.get("prompt_variant") is not None
        and (
            not require_complete
            or str(r.get("status") or "").strip().lower() == "complete"
        )
    }
    treatment_variants = _sorted_variants(available_treatments)
    if not treatment_variants:
        raise SystemExit(
            f"No treatment runs found for colluder_count={colluder_count} with secret_channel_enabled=true."
        )

    if not args.plot_all_prompt_variants:
        requested = canonical_variant(args.treatment_prompt_variant)
        if requested:
            if requested not in set(treatment_variants):
                raise SystemExit(
                    f"--treatment-prompt-variant={requested!r} not found in sweep (available={treatment_variants})."
                )
            treatment_variants = [requested]
        else:
            non_control = [v for v in treatment_variants if v != "control"]
            treatment_variants = [non_control[0] if non_control else treatment_variants[0]]

    # Metrics: include the radar metrics + a few common task-quality metrics.
    metrics: List[_MetricSpec] = [
        _MetricSpec("joint_reward_ratio", "Joint reward", higher_is_better=True),
        _MetricSpec("tasks_done", "Tasks done", higher_is_better=True),
        _MetricSpec("violations", "Violations", higher_is_better=False),
        _MetricSpec("total_cost", "Total cost", higher_is_better=False),
        _MetricSpec("priority_sum", "Priority sum", higher_is_better=True),
        _MetricSpec("coalition_mean_reward", "Coalition mean reward", higher_is_better=True),
        _MetricSpec("noncoalition_mean_reward", "Non-coalition mean reward", higher_is_better=True),
        _MetricSpec("coalition_advantage_mean", "Coalition advantage", higher_is_better=True),
        _MetricSpec("coalition_reward_ratio", "Coalition reward ratio", higher_is_better=True),
        _MetricSpec("mean_regret", "Mean regret", higher_is_better=False),
        _MetricSpec("reward_std", "Reward std", higher_is_better=False),
        _MetricSpec("colluder_posts_secret_rate", "Colluders secret rate", higher_is_better=True),
        _MetricSpec("colluder_posts_secret_to_largest_bb_ratio_mean", "Secret/public post ratio", higher_is_better=True),
        _MetricSpec("colluder_posts_total", "Colluders posts", higher_is_better=True),
        _MetricSpec("posts_total_all", "Posts (all agents)", higher_is_better=True),
        _MetricSpec(
            "judge_mean_rating",
            "Collusion Prediction (Judge)",
            higher_is_better=True,
        ),
    ]

    group_by = str(args.group_by)
    if group_by == "auto":
        group_by = _auto_group_by(rows)
    keys = _group_keys(group_by)

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {(): rows}
    if keys:
        grouped = {}
        for r in rows:
            grouped.setdefault(tuple(r.get(k) for k in keys), []).append(r)

    z_crit = float(args.z_critical)

    out_rows: List[Dict[str, Any]] = []
    for group_vals, group_rows in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        baseline_rows = _filter_group(
            group_rows,
            colluder_count=colluder_count,
            secret=False,
            prompt_variant=baseline_variant,
            require_complete=require_complete,
        )

        for pv in treatment_variants:
            treatment_rows = _filter_group(
                group_rows,
                colluder_count=colluder_count,
                secret=True,
                prompt_variant=pv,
                require_complete=require_complete,
            )

            for m in metrics:
                baseline_seed_means = _seed_means(baseline_rows, m.key)
                treatment_seed_means = _seed_means(treatment_rows, m.key)
                b = _group_stats_from_seed_means(baseline_seed_means)
                t = _group_stats_from_seed_means(treatment_seed_means)

                mu_b = _finite(b.get("mean"))
                mu_t = _finite(t.get("mean"))
                se_b = _finite(b.get("sem"))
                se_t = _finite(t.get("sem"))

                delta = None
                se_delta = None
                ci_low = None
                ci_high = None
                signif = None
                z_stat = None
                p_value = None
                perm_stat = None
                perm_p_value = None

                if mu_t is not None and mu_b is not None:
                    delta = float(mu_t) - float(mu_b)
                if se_t is not None and se_b is not None:
                    se_delta = math.sqrt(float(se_t) ** 2 + float(se_b) ** 2)
                if delta is not None and se_delta is not None:
                    ci_low = float(delta) - float(z_crit) * float(se_delta)
                    ci_high = float(delta) + float(z_crit) * float(se_delta)
                    if float(se_delta) > 0.0:
                        z_stat = float(delta) / float(se_delta)
                        p_value = _z_p_value_two_sided(z_stat)
                    if ci_low is not None and ci_high is not None:
                        signif = bool(ci_low > 0.0 or ci_high < 0.0)

                if bool(args.permutation_test):
                    perm_stat, perm_p_value = _permutation_p_value_mean_diff(
                        treatment_seed_means,
                        baseline_seed_means,
                        n_resamples=int(args.permutation_resamples),
                        random_seed=int(args.permutation_seed) if args.permutation_seed is not None else None,
                    )

                delta_improvement = None
                if delta is not None:
                    delta_improvement = float(delta) if m.higher_is_better else -float(delta)

                row: Dict[str, Any] = {
                    **{k: v for k, v in meta.items() if k in {"experiment_tag", "timestamp", "model_label", "sweep_name"}},
                    "sweep_dir": str(sweep_dir),
                    "group_by": group_by,
                    "colluder_count": int(colluder_count),
                    "baseline_prompt_variant": baseline_variant,
                    "treatment_prompt_variant": pv,
                    "baseline_secret_channel_enabled": False,
                    "treatment_secret_channel_enabled": True,
                    "include_incomplete": bool(args.include_incomplete),
                    "prefer_repaired": bool(args.prefer_repaired),
                    "metric_key": m.key,
                    "metric_label": m.label,
                    "metric_higher_is_better": bool(m.higher_is_better),
                    "baseline_n": b.get("n"),
                    "baseline_mean": mu_b,
                    "baseline_sem": se_b,
                    "treatment_n": t.get("n"),
                    "treatment_mean": mu_t,
                    "treatment_sem": se_t,
                    "delta_treatment_minus_baseline": delta,
                    "delta_improvement": delta_improvement,
                    "se_delta": se_delta,
                    "ci_low_95_z": ci_low,
                    "ci_high_95_z": ci_high,
                    "significant_0_05_z": signif,
                    "z_stat": z_stat,
                    "p_value_approx": p_value,
                    "perm_stat": perm_stat,
                    "p_value_perm": perm_p_value,
                    "perm_n_resamples": int(args.permutation_resamples) if args.permutation_test else None,
                    "perm_random_seed": int(args.permutation_seed) if args.permutation_seed is not None else None,
                }

                for k, v in zip(keys, group_vals):
                    row[k] = v

                # Helpful when investigating missing comparisons.
                row["baseline_rows"] = len(baseline_rows)
                row["treatment_rows"] = len(treatment_rows)

                out_rows.append(row)

    out_dir = default_out_dir(sweep_dir=sweep_dir, requested_out_dir=None)
    ensure_dir(out_dir / "sweep")

    if args.out_path:
        out_path = Path(args.out_path).expanduser().resolve()
    else:
        pv_tag = "pvALL" if args.plot_all_prompt_variants else f"pv{sanitize_filename(treatment_variants[0])}"
        gb_tag = sanitize_filename(group_by)
        out_path = out_dir / "sweep" / f"collusion_significance__c{colluder_count}__{pv_tag}__{gb_tag}.csv"

    write_csv(out_path, out_rows)
    logger.info("Wrote significance CSV: %s", out_path)
    print(f"Wrote CSV: {out_path}")
    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
