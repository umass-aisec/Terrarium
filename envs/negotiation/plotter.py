"""
Utility plotting module for Negotiation (Trading) Domain.
"""

import matplotlib.pyplot as plt
from typing import Dict, List
from envs.plotting_utils import (
    get_colors,
    apply_plot_styling,
    create_legend,
    save_plot,
    generate_plot_filepath,
    set_integer_xticks
)


class UtilityPlotter:
    """
    Handles plotting of agent utilities and budgets over time for negotiation environments.
    Uses dual y-axis to track both utility (left) and budget (right).
    """

    def __init__(self, save_dir: str = "plots"):
        """
        Initialize the utility plotter.

        Args:
            save_dir: Directory to save plots (created if doesn't exist)
        """
        self.save_dir = save_dir
        self.colors = get_colors()

    def plot_utilities(self, utility_history: Dict[str, List[int]],
                      iteration: int, budget_history: Dict[str, List[int]] = None,
                      show: bool = False) -> str:
        """
        Plot utilities after each iteration with custom formatting and budget tracking.

        Args:
            utility_history: Dictionary mapping agent names to utility lists
            iteration: Current iteration number
            budget_history: Dictionary mapping agent names to budget lists (optional)
            show: Whether to display the plot interactively

        Returns:
            Path to saved plot file
        """

        # Create 25x15 figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(20, 15))
        ax2 = ax1.twinx()  # Create secondary y-axis for budget

        # Collect handles and labels for combined legend with agent names for sorting
        utility_data = []  # List of (agent_name, handle)
        budget_data = []   # List of (agent_name, handle)

        # Plot each agent's utility history (left y-axis) and budget history (right y-axis)
        for idx, (agent_name, utilities) in enumerate(utility_history.items()):
            if utilities:
                iterations = list(range(1, len(utilities) + 1))
                color = self.colors[idx % len(self.colors)]

                # Use agent name as display name
                display_name = agent_name

                # Plot utilities (solid line) - increased sizes by 30%
                utility_line, = ax1.plot(iterations, utilities,
                                       marker='o', label=f"{display_name} (Utility)",
                                       color=color, linewidth=3.9, markersize=13,
                                       markeredgecolor='white', markeredgewidth=2.0)
                utility_data.append((agent_name, utility_line))

                # Plot budgets if available (dashed line)
                if budget_history and agent_name in budget_history and budget_history[agent_name]:
                    budgets = budget_history[agent_name][:len(iterations)]  # Match length
                    budget_line, = ax2.plot(iterations, budgets,
                                          linestyle='--', label=f"{display_name} (Budget)",
                                          color=color, linewidth=3.9,
                                          alpha=0.56)
                    budget_data.append((agent_name, budget_line))

        # Apply consistent styling
        title = f'Agent Utilities & Budgets - Iteration {iteration}'
        apply_plot_styling(ax1, title, 'Iteration', 'Utility', 'Budget', ax2)

        # Create two-row legend below plot with utility on top, budget on bottom
        # Sort handles by agent name to ensure consistent ordering
        utility_data.sort(key=lambda x: x[0])  # Sort by agent name
        budget_data.sort(key=lambda x: x[0])   # Sort by agent name

        # Extract sorted handles
        utility_handles = [handle for _, handle in utility_data]
        budget_handles = [handle for _, handle in budget_data]

        num_agents = len(utility_handles)

        # Ensure we have matching entries for all agents (pad with None if needed)
        while len(budget_handles) < num_agents:
            budget_handles.append(None)

        # Create interleaved handles for proper row-wise display
        # For row-wise arrangement, we need to interleave utilities and budgets
        all_handles = []
        all_labels = []

        # First row: all utilities
        for handle in utility_handles:
            all_handles.append(handle)
            all_labels.append(handle.get_label())

        # Second row: all budgets
        for handle in budget_handles:
            if handle is not None:
                all_handles.append(handle)
                all_labels.append(handle.get_label())

        # Force 2 rows by setting ncol to half the total items (rounded up)
        total_items = len(all_handles)
        ncols = (total_items + 1) // 2  # This ensures 2 rows

        create_legend(ax1, all_handles, all_labels, 'Agent Metrics', ncols)

        # Ensure integer x-axis ticks for iterations
        if utility_history:
            max_iter = max(len(utils) for utils in utility_history.values() if utils)
            set_integer_xticks(ax1, max_iter)

        # Save plot with timestamp
        filepath = generate_plot_filepath(self.save_dir, "utility_plot", iteration)
        save_plot(fig, filepath)

        if show:
            plt.show()
        else:
            plt.close()

        return filepath
