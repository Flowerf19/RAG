"""
Table Output Utilities for RAG Metrics Visualization
====================================================

Functions to generate and display metrics tables in various formats.
"""

import pandas as pd
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_metrics_table(df: pd.DataFrame,
                          title: str = "RAG Metrics Comparison",
                          output_format: str = "markdown",
                          save_path: Optional[Union[str, Path]] = None) -> str:
    """
    Generate a formatted table from metrics DataFrame.

    Args:
        df: DataFrame with metrics data
        title: Title for the table
        output_format: Output format ('markdown', 'html', 'latex', 'csv')
        save_path: Optional path to save the table

    Returns:
        Formatted table string
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for table generation")
            return "No data available"

        # Format numeric columns to 4 decimal places
        formatted_df = df.copy()
        numeric_cols = ['Faithfulness', 'Context_Recall', 'Context_Relevance', 'Answer_Relevancy']
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].round(4)

        # Generate table based on format
        if output_format == "markdown":
            table_str = f"## {title}\n\n"
            table_str += formatted_df.to_markdown(index=False)
        elif output_format == "html":
            table_str = f"<h2>{title}</h2>\n"
            table_str += formatted_df.to_html(index=False, classes='table table-striped')
        elif output_format == "latex":
            table_str = f"\\section{{{title}}}\n\n"
            table_str += formatted_df.to_latex(index=False, float_format="%.4f")
        elif output_format == "csv":
            table_str = formatted_df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Save to file if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(table_str)
            logger.info(f"Table saved to {save_path}")

        return table_str

    except Exception as e:
        logger.error(f"Error generating metrics table: {e}")
        return f"Error generating table: {e}"


def print_metrics_table(df: pd.DataFrame,
                       title: str = "RAG Metrics Comparison") -> None:
    """
    Print metrics table to console.

    Args:
        df: DataFrame with metrics data
        title: Title for the table
    """
    try:
        table_str = generate_metrics_table(df, title, output_format="markdown")
        print(table_str)
        print()  # Add blank line after table

    except Exception as e:
        logger.error(f"Error printing metrics table: {e}")
        print(f"Error printing table: {e}")


def save_metrics_table(df: pd.DataFrame,
                      save_path: Union[str, Path],
                      title: str = "RAG Metrics Comparison",
                      output_format: str = "markdown") -> bool:
    """
    Save metrics table to file.

    Args:
        df: DataFrame with metrics data
        save_path: Path to save the table
        title: Title for the table
        output_format: Output format

    Returns:
        True if successful, False otherwise
    """
    try:
        generate_metrics_table(df, title, output_format, save_path)
        logger.info(f"Metrics table saved to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving metrics table: {e}")
        return False