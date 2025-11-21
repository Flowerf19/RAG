"""
Heatmap for RAG Metrics Visualization
====================================

Implementation of heatmap to visualize RAG metrics across configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Optional, Union, Tuple
from pathlib import Path

from ..utils.save_utils import save_chart

logger = logging.getLogger(__name__)


def generate_heatmap(df: pd.DataFrame,
                    title: str = "RAG Metrics Heatmap",
                    figsize: Tuple[int, int] = (10, 6),
                    cmap: str = "YlGnBu",
                    save_path: Optional[Union[str, Path]] = None,
                    show_plot: bool = True) -> plt.Figure:
    """
    Generate heatmap for RAG metrics visualization.

    Args:
        df: DataFrame with columns: Configuration, Faithfulness, Context_Recall, Answer_Correctness, Answer_Relevancy
        title: Chart title
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        save_path: Optional path to save chart
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for heatmap")
            return plt.figure()

        # Prepare data for heatmap
        # Set Configuration as index, keep only metric columns
        heatmap_data = df.set_index('Configuration')[
            ['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']
        ]

        # Rename columns for better display
        heatmap_data.columns = ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy']

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Generate heatmap
        sns.heatmap(heatmap_data,
                   annot=True,
                   fmt='.3f',
                   cmap=cmap,
                   ax=ax,
                   cbar_kws={'label': 'Score (0-1)'},
                   linewidths=0.5,
                   square=False)

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save chart if requested
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Heatmap saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return plt.figure()


def generate_heatmap_with_clustering(df: pd.DataFrame,
                                    title: str = "RAG Metrics Heatmap with Clustering",
                                    figsize: Tuple[int, int] = (12, 8),
                                    cmap: str = "RdYlGn",
                                    save_path: Optional[Union[str, Path]] = None,
                                    show_plot: bool = True) -> plt.Figure:
    """
    Generate clustered heatmap for advanced analysis of metrics patterns.

    Args:
        df: DataFrame with metrics data
        title: Chart title
        figsize: Figure size
        cmap: Colormap
        save_path: Optional save path
        show_plot: Whether to display

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for clustered heatmap")
            return plt.figure()

        # Prepare data
        heatmap_data = df.set_index('Configuration')[
            ['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']
        ]
        heatmap_data.columns = ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy']

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Generate clustered heatmap
        sns.clustermap(heatmap_data,
                      annot=True,
                      fmt='.3f',
                      cmap=cmap,
                      figsize=figsize,
                      cbar_pos=(0.02, 0.32, 0.05, 0.45))

        # Set title
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        # Save chart if requested (clustermap creates its own figure)
        if save_path:
            # For clustermap, we need to save the current figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Clustered heatmap saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        return plt.gcf()

    except Exception as e:
        logger.error(f"Error generating clustered heatmap: {e}")
        return plt.figure()


def generate_difference_heatmap(df: pd.DataFrame,
                               baseline_config: str = None,
                               title: str = "RAG Metrics Difference from Baseline",
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[Union[str, Path]] = None,
                               show_plot: bool = True) -> plt.Figure:
    """
    Generate heatmap showing differences from a baseline configuration.

    Args:
        df: DataFrame with metrics data
        baseline_config: Name of baseline configuration (first one if None)
        title: Chart title
        figsize: Figure size
        save_path: Optional save path
        show_plot: Whether to display

    Returns:
        Matplotlib figure object
    """
    try:
        if df.empty or len(df) < 2:
            logger.warning("Need at least 2 configurations for difference heatmap")
            return plt.figure()

        # Set baseline
        if baseline_config is None:
            baseline_config = df['Configuration'].iloc[0]

        # Get baseline values
        baseline_row = df[df['Configuration'] == baseline_config]
        if baseline_row.empty:
            logger.warning(f"Baseline configuration '{baseline_config}' not found")
            return plt.figure()

        baseline_values = baseline_row[['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']].iloc[0]

        # Calculate differences
        diff_data = df.set_index('Configuration')[['Faithfulness', 'Context_Recall', 'Answer_Correctness', 'Answer_Relevancy']].copy()
        for col in diff_data.columns:
            diff_data[col] = diff_data[col] - baseline_values[col]

        # Rename columns
        diff_data.columns = ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy']

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Generate difference heatmap
        sns.heatmap(diff_data,
                   annot=True,
                   fmt='.3f',
                   cmap='RdBu_r',  # Red-Blue diverging colormap
                   ax=ax,
                   center=0,
                   cbar_kws={'label': f'Difference from {baseline_config}'},
                   linewidths=0.5)

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save and show
        if save_path:
            success = save_chart(fig, save_path.stem if hasattr(save_path, 'stem') else str(save_path).replace('.png', ''),
                               save_path.parent if hasattr(save_path, 'parent') else Path('.'))
            if success:
                logger.info(f"Difference heatmap saved to {save_path}")

        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error generating difference heatmap: {e}")
        return plt.figure()