"""
Shared plotting utilities for visualization across different environments.

This module provides common plotting functions used by both negotiation and DCOP
environment plotters to ensure consistent styling and reduce code duplication.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from typing import List, Tuple, Optional
import os
from datetime import datetime


def get_colors() -> List[str]:
    """
    Return standardized color palette for consistent plotting across environments.

    Returns:
        List of hex color codes (professional academic color palette)
    """
    return [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]

def apply_plot_styling(ax: Axes, title: str, xlabel: str, ylabel: str,
                       ylabel2: Optional[str] = None, ax2: Optional[Axes] = None) -> None:
    """
    Apply consistent styling to plot axes including fonts, grid, and background.

    Args:
        ax: Primary matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label (primary)
        ylabel2: Y-axis label for secondary axis (optional)
        ax2: Secondary matplotlib axis (optional, for dual-axis plots)
    """
    # Set font sizes and labels for primary axis
    ax.set_xlabel(xlabel, fontsize=21)
    ax.set_ylabel(ylabel, fontsize=21, color='black')
    ax.set_title(title, fontsize=26, pad=20)

    # Set tick parameters for primary axis
    ax.tick_params(axis='both', which='major', labelsize=21)

    # Handle secondary axis if provided
    if ax2 is not None and ylabel2 is not None:
        ax2.set_ylabel(ylabel2, fontsize=21, color='black')
        ax2.tick_params(axis='y', which='major', labelsize=21)

    # Add grid with custom styling (only on primary axis)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set background colors
    ax.set_facecolor('#F8F8F8')
    ax.figure.set_facecolor('white')

    # Add minor gridlines for better readability
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)


def create_legend(ax: Axes, handles: List, labels: List, title: str,
                 ncols: Optional[int] = None) -> Legend:
    """
    Create a standardized legend with consistent formatting.

    Args:
        ax: Matplotlib axis to attach legend to
        handles: List of plot handles (lines, etc.)
        labels: List of labels corresponding to handles
        title: Legend title
        ncols: Number of columns (defaults to aim for 2 rows)

    Returns:
        Created legend object
    """
    # Calculate number of columns if not provided (aim for 2 rows max)
    if ncols is None:
        total_items = len(handles)
        ncols = (total_items + 1) // 2 if total_items > 2 else total_items

    # Create legend with standard formatting
    legend = ax.legend(handles, labels, fontsize=18,
                      bbox_to_anchor=(0.5, -0.15), loc='upper center',
                      ncol=ncols, framealpha=0.9, edgecolor='gray',
                      fancybox=True)
    legend.set_title(title)
    legend.get_title().set_fontsize(23)

    return legend


def save_plot(fig: Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save plot with consistent settings and cleanup.

    Args:
        fig: Matplotlib figure to save
        filepath: Full path where plot should be saved
        dpi: Resolution (dots per inch) for saved image
    """
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')


def generate_plot_filepath(save_dir: str, prefix: str, iteration: Optional[int] = None) -> str:
    """
    Generate standardized filepath for plot with timestamp.

    Args:
        save_dir: Directory to save plot in
        prefix: Prefix for filename (e.g., "utility_plot", "scores")
        iteration: Optional iteration number to include in filename

    Returns:
        Full filepath string
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename based on whether iteration is provided
    if iteration is not None:
        filename = f"{prefix}_iter_{iteration}_{timestamp}.png"
    else:
        filename = f"{prefix}_final_{timestamp}.png"

    return os.path.join(save_dir, filename)


def set_integer_xticks(ax: Axes, max_iter: int) -> None:
    """
    Set x-axis ticks to integers from 1 to max_iter.

    Args:
        ax: Matplotlib axis to configure
        max_iter: Maximum iteration number
    """
    if max_iter > 0:
        ax.set_xticks(range(1, max_iter + 1))
