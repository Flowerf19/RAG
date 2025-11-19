"""
Grouped Bar Chart for RAG Metrics Visualization
===============================================

Implementation of grouped bar chart to compare RAG metrics across configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Optional, Union, Tuple
from pathlib import Path

from ..utils.save_utils import save_chart

logger = logging.getLogger(__name__)


def generate_bar_chart(df: pd.DataFrame,
                      title: str = "RAG Metrics Comparison Across Configurations",
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[Union[str, Path]] = None,
                      show_plot: bool = True) -> plt.Figure:
    """
    Generate grouped bar chart for RAG metrics comparison.

    Args:
        df: DataFrame with columns: Configuration, Faithfulness, Context_Recall, Context_Relevance, Answer_Relevancy
        title: Chart title
        figsize: Figure size (width, height)
        save_path: Optional path to save chart
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for bar chart")
            return plt.figure()

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

        # Prepare data for plotting
        metrics = ['Faithfulness', 'Context_Recall', 'Context_Relevance', 'Answer_Relevancy']
        metric_labels = ['Faithfulness', 'Context Recall', 'Context Relevance', 'Answer Relevancy']

        # Melt dataframe for seaborn
        df_melted = df.melt(id_vars=['Configuration'],
                           value_vars=metrics,
                           var_name='Metric',
                           value_name='Score')

        # Map metric names to display labels
        metric_display_map = dict(zip(metrics, metric_labels))
        df_melted['Metric'] = df_melted['Metric'].map(metric_display_map)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create grouped bar chart
        sns.barplot(data=df_melted,
                   x='Metric',
                   y='Score',
                   hue='Configuration',
                   ax=ax,
                   palette='Set2')

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Score (0-1)', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylim(0, 1.1)

        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)

        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

        # Adjust legend
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save chart if requested
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Bar chart saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating bar chart: {e}")
        return plt.figure()


def generate_horizontal_bar_chart(df: pd.DataFrame,
                                 title: str = "RAG Metrics Comparison (Horizontal)",
                                 figsize: Tuple[int, int] = (10, 8),
                                 save_path: Optional[Union[str, Path]] = None,
                                 show_plot: bool = True) -> plt.Figure:
    """
    Generate horizontal grouped bar chart for better readability with many configurations.

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
            logger.warning("Empty DataFrame provided for horizontal bar chart")
            return plt.figure()

        # Set style
        sns.set_style("whitegrid")

        # Prepare data
        metrics = ['Faithfulness', 'Context_Recall', 'Context_Relevance', 'Answer_Relevancy']
        metric_labels = ['Faithfulness', 'Context Recall', 'Context Relevance', 'Answer Relevancy']

        df_melted = df.melt(id_vars=['Configuration'],
                           value_vars=metrics,
                           var_name='Metric',
                           value_name='Score')

        metric_display_map = dict(zip(metrics, metric_labels))
        df_melted['Metric'] = df_melted['Metric'].map(metric_display_map)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Horizontal bar chart
        sns.barplot(data=df_melted,
                   y='Metric',
                   x='Score',
                   hue='Configuration',
                   ax=ax,
                   palette='Set2',
                   orient='h')

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Score (0-1)', fontsize=12)
        ax.set_ylabel('Metrics', fontsize=12)
        ax.set_xlim(0, 1.1)

        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

        # Adjust legend
        ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # Save and show
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Horizontal bar chart saved to {save_path}")

        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating horizontal bar chart: {e}")
        return plt.figure()