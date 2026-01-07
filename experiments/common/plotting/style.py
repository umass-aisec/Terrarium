from __future__ import annotations

from typing import Any


def apply_default_style(plt: Any) -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-v0_8", "ggplot"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
