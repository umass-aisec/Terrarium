"""
Example health metric hook for the Terrarium dashboard.

Usage:
    python dashboards/build_data.py --logs-root logs \\
        --health-metric dashboards/health_metric.py \\
        --output dashboards/dashboard_v2/dashboard_data.json
"""
from __future__ import annotations

from typing import Any, Dict


def compute_health(run: Dict[str, Any]) -> Dict[str, Any]:
    """Return a health annotation for a run.

    Expected shape:
        {
            "ok": bool,           # True means healthy, False means needs attention
            "label": str,         # Short badge text
            "reason": str,        # Optional longer explanation
            "score": float,       # Optional numeric score (0-100 recommended)
        }
    """
    # Prefer an explicit success_rate if present
    success_rate = run.get("success_rate")
    if isinstance(success_rate, (int, float)):
        ok = success_rate >= 50
        return {
            "ok": ok,
            "label": "Likely healthy" if ok else "Needly attention",
            "reason": f"Success rate: {success_rate:.1f}%",
            "score": float(success_rate),
        }

    # Fallback to event_counts ratio
    counts = run.get("event_counts") or {}
    success = sum((counts.get(k) or {}).get("success", 0) for k in counts)
    failure = sum((counts.get(k) or {}).get("failure", 0) for k in counts)
    total = success + failure
    if total > 0:
        rate = (success / total) * 100
        ok = rate >= 50
        return {
            "ok": ok,
            "label": "Likely healthy" if ok else "Needly attention",
            "reason": f"{success}/{total} successful events ({rate:.1f}%)",
            "score": float(rate),
        }

    # No signal yet: stay neutral
    return {
        "ok": False,
        "label": "Needly attention",
        "reason": "No event data yet; awaiting signal.",
        "score": 0.0,
    }
