from __future__ import annotations

from typing import Any


def apply_default_style(plt: Any) -> None:
    try:
        plt.style.use("default")
    except Exception:
        pass
    # Aim for paper-friendly, text-consistent defaults.
    # If you change `font_size`, keep the other font-related sizes aligned unless you
    # intentionally want hierarchy (titles > labels > ticks).
    font_size = 26
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
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    )
