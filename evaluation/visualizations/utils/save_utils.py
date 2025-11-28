"""
Save Utilities for RAG Metrics Visualization
============================================

Functions to save charts and visualizations to files.
"""

import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def save_chart(fig: plt.Figure,
              filename: str,
              output_dir: Union[str, Path] = "data/visualizations",
              dpi: int = 300,
              bbox_inches: str = 'tight') -> bool:
    """
    Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure to save
        filename: Filename (without extension)
        output_dir: Directory to save to
        dpi: Resolution for PNG
        bbox_inches: Bounding box setting

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / f"{filename}.png"

        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        logger.info(f"Chart saved to {filepath}")

        return True

    except Exception as e:
        logger.error(f"Error saving chart: {e}")
        return False


def setup_visualization_directory(base_dir: Union[str, Path] = "data/visualizations") -> Path:
    """
    Ensure visualization directory exists and return Path object.

    Args:
        base_dir: Base directory for visualizations

    Returns:
        Path to visualization directory
    """
    try:
        viz_dir = Path(base_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization directory ready: {viz_dir}")
        return viz_dir

    except Exception as e:
        logger.error(f"Error setting up visualization directory: {e}")
        return Path("data/visualizations")  # Fallback


def get_default_save_path(filename: str,
                         base_dir: Union[str, Path] = "data/visualizations") -> Path:
    """
    Get default save path for a chart file.

    Args:
        filename: Filename without extension
        base_dir: Base directory

    Returns:
        Full path for the file
    """
    viz_dir = setup_visualization_directory(base_dir)
    return viz_dir / f"{filename}.png"