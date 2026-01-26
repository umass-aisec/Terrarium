from __future__ import annotations

from typing import Any


def apply_default_style(plt: Any) -> None:
    try:
        plt.style.use("default")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    )
