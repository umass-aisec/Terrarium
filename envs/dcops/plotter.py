"""
Score plotting module for DCOP environments.
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


class ScorePlotter:
    """
    Handles plotting of global and local scores over time for DCOP environments.
    Uses single y-axis to track both global and local scores.
    """

    def __init__(self, save_dir: str = "plots"):
        """
        Initialize the score plotter.

        Args:
            save_dir: Directory to save plots (created if doesn't exist)
        """
        self.save_dir = save_dir
        self.colors = get_colors()

    def plot_scores(self, global_score_history: List[float],
                   local_scores_history: Dict[str, List[float]],
                   iteration: int, environment_name: str = "Unknown",
                   show: bool = False) -> str:
        """
        Plot global and local scores after each iteration.

        Args:
            global_score_history: List of global scores over iterations
            local_scores_history: Dictionary mapping agent names to local score lists
            iteration: Current iteration number
            environment_name: Name of the environment for labeling
            show: Whether to display the plot interactively

        Returns:
            Path to saved plot file
        """

        # Create 25x15 figure with single y-axis
        fig, ax1 = plt.subplots(figsize=(20, 15))

        # Plot global score history - bold line
        if global_score_history:
            iterations = list(range(1, len(global_score_history) + 1))
            global_line, = ax1.plot(iterations, global_score_history,
                                  marker='s', label="Global Score",
                                  color='#000080', linewidth=5.2, markersize=15,
                                  markeredgecolor='white', markeredgewidth=2.5)

        # Collect handles and labels for combined legend with agent names for sorting
        local_data = []  # List of (agent_name, handle)

        # Plot each agent's local score history
        for idx, (agent_name, scores) in enumerate(local_scores_history.items()):
            if scores:
                iterations = list(range(1, len(scores) + 1))
                color = self.colors[idx % len(self.colors)]

                # Plot local scores (solid line) - increased sizes by 30%
                local_line, = ax1.plot(iterations, scores,
                                     marker='o', label=f"{agent_name} (Local)",
                                     color=color, linewidth=3.9, markersize=13,
                                     markeredgecolor='white', markeredgewidth=2.0)
                local_data.append((agent_name, local_line))

        title = f'{environment_name} - Global & Local Scores - Iteration {iteration}'
        apply_plot_styling(ax1, title, 'Iteration', 'Score')

        # Create legend with global score first, then sorted local scores
        all_handles = []
        all_labels = []

        # Add global score first
        if global_score_history:
            all_handles.append(global_line)
            all_labels.append(global_line.get_label())

        # Sort local scores by agent name and add them
        local_data.sort(key=lambda x: x[0])  # Sort by agent name
        for _, handle in local_data:
            all_handles.append(handle)
            all_labels.append(handle.get_label())

        # Calculate number of columns for legend (aim for 2 rows max)
        total_items = len(all_handles)
        ncols = (total_items + 1) // 2 if total_items > 2 else total_items

        create_legend(ax1, all_handles, all_labels, 'Score Metrics', ncols)

        # Ensure integer x-axis ticks for iterations
        if global_score_history or local_scores_history:
            max_iter = max(
                len(global_score_history) if global_score_history else 0,
                max(len(scores) for scores in local_scores_history.values() if scores) if local_scores_history else 0
            )
            set_integer_xticks(ax1, max_iter)

        # Save plot with timestamp
        filepath = generate_plot_filepath(self.save_dir, "scores", iteration)
        save_plot(fig, filepath)

        if show:
            plt.show()
        else:
            plt.close()

        return filepath
