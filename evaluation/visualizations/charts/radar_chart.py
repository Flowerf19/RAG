"""
Radar Chart for RAG Metrics Visualization
=========================================

Implementation of radar/spider chart to visualize RAG metrics profiles across configurations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from math import pi
from typing import Optional, Union, Tuple
from pathlib import Path

from ..utils.save_utils import save_chart

logger = logging.getLogger(__name__)


def generate_radar_chart(df: pd.DataFrame,
                        title: str = "RAG Metrics Profile Comparison",
                        figsize: Tuple[int, int] = (8, 8),
                        save_path: Optional[Union[str, Path]] = None,
                        show_plot: bool = True) -> plt.Figure:
    """
    Generate radar chart for RAG metrics profile comparison.

    Args:
        df: DataFrame with columns: Configuration, Faithfulness, Context_Recall, Answer_Correctness, Answer_Relevancy
        title: Chart title
        figsize: Figure size (width, height)
        save_path: Optional path to save chart
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for radar chart")
            return plt.figure()

        # Prepare data
        metrics = ['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']
        metric_labels = ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy']

        # Number of variables
        N = len(metrics)

        # Angle for each metric
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Create figure with polar axes
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # Plot each configuration
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

        for i, (_, row) in enumerate(df.iterrows()):
            config_name = row['Configuration']
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the loop

            # Plot the line
            ax.plot(angles, values, 'o-', linewidth=2, label=config_name, color=colors[i])

            # Fill the area
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Add labels for each metric
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)

        # Set radial limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add title
        ax.set_title(title, size=14, fontweight='bold', pad=20)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()

        # Save chart if requested
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Radar chart saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating radar chart: {e}")
        return plt.figure()


def generate_radar_chart_subplots(df: pd.DataFrame,
                                 title: str = "RAG Metrics Profiles",
                                 figsize: Tuple[int, int] = (12, 8),
                                 save_path: Optional[Union[str, Path]] = None,
                                 show_plot: bool = True) -> plt.Figure:
    """
    Generate radar charts in subplots for better comparison when many configurations.

    Args:
        df: DataFrame with metrics data
        title: Chart title
        figsize: Figure size
        save_path: Optional save path
        show_plot: Whether to display

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for radar subplots")
            return plt.figure()

        # Prepare data
        metrics = ['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']
        metric_labels = ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy']

        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Calculate subplot grid
        n_configs = len(df)
        n_cols = min(3, n_configs)
        n_rows = (n_configs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                subplot_kw=dict(polar=True))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Flatten axes array for easy iteration
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        # Plot each configuration
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]
            config_name = row['Configuration']
            values = [row[metric] for metric in metrics]
            values += values[:1]

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.3, color='blue')

            # Labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.5, 1.0])
            ax.set_yticklabels(['0.5', '1.0'], fontsize=6)
            ax.grid(True, alpha=0.3)

            # Title for subplot
            ax.set_title(config_name, fontsize=10, pad=10)

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()

        # Save and show
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Radar subplots saved to {save_path}")

        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating radar subplots: {e}")
        return plt.figure()