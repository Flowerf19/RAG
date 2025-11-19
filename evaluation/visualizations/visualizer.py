"""
RAG Metrics Visualizer
======================

Main orchestrator for generating RAG evaluation visualizations.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .utils.data_prep import prepare_metrics_dataframe, prepare_metrics_from_ragas_output
from .utils.table_output import generate_metrics_table, print_metrics_table
from .charts.bar_chart import generate_bar_chart, generate_horizontal_bar_chart
from .charts.radar_chart import generate_radar_chart, generate_radar_chart_subplots
from .charts.heatmap import generate_heatmap, generate_difference_heatmap

logger = logging.getLogger(__name__)


class RAGMetricsVisualizer:
    """
    Main class for generating RAG metrics visualizations.
    """

    def __init__(self, output_dir: Union[str, Path] = "data/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RAGMetricsVisualizer initialized with output dir: {self.output_dir}")

    def generate_all_charts(self,
                           df: pd.DataFrame,
                           title_prefix: str = "RAG Evaluation",
                           save_charts: bool = True,
                           show_charts: bool = False) -> Dict[str, Any]:
        """
        Generate all available chart types for the given metrics data.

        Args:
            df: DataFrame with metrics data
            title_prefix: Prefix for chart titles
            save_charts: Whether to save charts to files
            show_charts: Whether to display charts

        Returns:
            Dictionary with chart file paths and status
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return {"error": "Empty DataFrame"}

            results = {}

            # Generate table first
            table_title = f"{title_prefix} - Metrics Table"
            print_metrics_table(df, table_title)

            if save_charts:
                table_file = self.output_dir / "metrics_table.md"
                generate_metrics_table(df, table_title, output_format="markdown", save_path=table_file)
                results["table"] = str(table_file)

            # Generate bar chart
            bar_title = f"{title_prefix} - Bar Chart Comparison"
            fig_bar = generate_bar_chart(df, bar_title, show_plot=show_charts)
            if save_charts:
                bar_file = self.output_dir / "bar_chart.png"
                fig_bar.savefig(bar_file, dpi=300, bbox_inches='tight', facecolor='white')
                results["bar_chart"] = str(bar_file)

            # Generate horizontal bar chart if many configs
            if len(df) > 3:
                hbar_title = f"{title_prefix} - Horizontal Bar Chart"
                fig_hbar = generate_horizontal_bar_chart(df, hbar_title, show_plot=show_charts)
                if save_charts:
                    hbar_file = self.output_dir / "horizontal_bar_chart.png"
                    fig_hbar.savefig(hbar_file, dpi=300, bbox_inches='tight', facecolor='white')
                    results["horizontal_bar_chart"] = str(hbar_file)

            # Generate radar chart
            radar_title = f"{title_prefix} - Radar Profile Comparison"
            if len(df) <= 5:  # Single radar for few configs
                fig_radar = generate_radar_chart(df, radar_title, show_plot=show_charts)
                if save_charts:
                    radar_file = self.output_dir / "radar_chart.png"
                    fig_radar.savefig(radar_file, dpi=300, bbox_inches='tight', facecolor='white')
                    results["radar_chart"] = str(radar_file)
            else:  # Subplots for many configs
                fig_radar = generate_radar_chart_subplots(df, radar_title, show_plot=show_charts)
                if save_charts:
                    radar_file = self.output_dir / "radar_subplots.png"
                    fig_radar.savefig(radar_file, dpi=300, bbox_inches='tight', facecolor='white')
                    results["radar_chart"] = str(radar_file)

            # Generate heatmap
            heatmap_title = f"{title_prefix} - Metrics Heatmap"
            fig_heatmap = generate_heatmap(df, heatmap_title, show_plot=show_charts)
            if save_charts:
                heatmap_file = self.output_dir / "heatmap.png"
                fig_heatmap.savefig(heatmap_file, dpi=300, bbox_inches='tight', facecolor='white')
                results["heatmap"] = str(heatmap_file)

            # Generate difference heatmap if multiple configs
            if len(df) > 1:
                diff_title = f"{title_prefix} - Difference from Baseline"
                fig_diff = generate_difference_heatmap(df, title=diff_title, show_plot=show_charts)
                if save_charts:
                    diff_file = self.output_dir / "difference_heatmap.png"
                    fig_diff.savefig(diff_file, dpi=300, bbox_inches='tight', facecolor='white')
                    results["difference_heatmap"] = str(diff_file)

            logger.info(f"Generated {len(results)} visualizations in {self.output_dir}")
            return results

        except Exception as e:
            logger.error(f"Error generating all charts: {e}")
            return {"error": str(e)}

    def visualize_from_ragas_output(self,
                                   ragas_summary: Dict[str, Any],
                                   config_name: str = "Current_Config",
                                   title_prefix: str = "RAG Evaluation",
                                   save_charts: bool = True,
                                   show_charts: bool = False) -> Dict[str, Any]:
        """
        Generate visualizations directly from Ragas evaluation output.

        Args:
            ragas_summary: Summary dict from Ragas evaluation
            config_name: Name for this configuration
            title_prefix: Prefix for titles
            save_charts: Whether to save charts
            show_charts: Whether to show charts

        Returns:
            Dictionary with results
        """
        try:
            df = prepare_metrics_from_ragas_output(ragas_summary, config_name)
            if df.empty:
                return {"error": "Failed to prepare DataFrame from Ragas output"}

            return self.generate_all_charts(df, title_prefix, save_charts, show_charts)

        except Exception as e:
            logger.error(f"Error visualizing from Ragas output: {e}")
            return {"error": str(e)}

    def compare_configurations(self,
                              evaluation_results: List[Dict[str, Any]],
                              config_names: Optional[List[str]] = None,
                              title_prefix: str = "RAG Configuration Comparison",
                              save_charts: bool = True,
                              show_charts: bool = False) -> Dict[str, Any]:
        """
        Compare multiple RAG configurations.

        Args:
            evaluation_results: List of evaluation result dictionaries
            config_names: Optional names for configurations
            title_prefix: Prefix for titles
            save_charts: Whether to save charts
            show_charts: Whether to show charts

        Returns:
            Dictionary with results
        """
        try:
            df = prepare_metrics_dataframe(evaluation_results, config_names)
            if df.empty:
                return {"error": "Failed to prepare DataFrame from evaluation results"}

            return self.generate_all_charts(df, title_prefix, save_charts, show_charts)

        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
            return {"error": str(e)}

    def create_sample_visualization(self,
                                   save_charts: bool = True,
                                   show_charts: bool = False) -> Dict[str, Any]:
        """
        Create sample visualizations with mock data for testing.

        Returns:
            Dictionary with results
        """
        # Sample data
        sample_data = {
            'Configuration': [
                'Local + No Re-ranking',
                'API + Query Rewrite + Re-ranking',
                'Local + API Hybrid + No Rewrite'
            ],
            'Faithfulness': [1.000, 0.950, 0.980],
            'Context_Recall': [1.000, 0.920, 0.990],
            'Context_Relevance': [1.000, 0.940, 0.970],
            'Answer_Relevancy': [1.000, 0.960, 0.985]
        }
        df = pd.DataFrame(sample_data)

        return self.generate_all_charts(df, "Sample RAG Evaluation", save_charts, show_charts)