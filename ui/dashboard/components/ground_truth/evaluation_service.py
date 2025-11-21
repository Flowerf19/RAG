"""
Ground Truth Evaluation Service
Handles the full evaluation suite logic extracted from GroundTruthComponent.
"""

import streamlit as st
import pandas as pd
import logging


class GroundTruthEvaluationService:
    """Service for running ground truth evaluations."""

    def __init__(self, backend, logger=None):
        self.backend = backend
        self.logger = logger or logging.getLogger(__name__)

    def run_full_evaluation_suite(self, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db):
        """Run complete evaluation suite with Ragas metrics."""
        try:
            self.logger.info("Starting Ragas evaluation suite: embedder=%s reranker=%s llm=%s use_qem=%s sample_size=%s save_to_db=%s",
                             embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db)
        except Exception:
            pass

        with st.spinner("Running Ragas evaluation suite (may take several minutes)..."):
            try:
                # Run Ragas evaluation with all metrics
                ragas_result = self.backend.evaluate_ground_truth_with_ragas(
                    llm_provider=llm_choice if llm_choice != "none" else "ollama",
                    embedder_choice=embedder_choice,
                    reranker_choice=reranker_choice,
                    limit=None if sample_size == 0 else sample_size,
                    save_to_db=save_to_db,
                )

                # Display comprehensive summary
                st.subheader("Ragas Evaluation Suite Results")

                # Create comparison table
                summary = ragas_result.get('summary', {})
                comparison_data = {
                    'Metric': ['Faithfulness', 'Context Recall', 'Answer Correctness', 'Answer Relevancy'],
                    'Score': [
                        f"{summary.get('faithfulness', 0):.4f}",
                        f"{summary.get('context_recall', 0):.4f}",
                        f"{summary.get('answer_correctness', 0):.4f}",
                        f"{summary.get('answer_relevancy', 0):.4f}"
                    ]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)

                # Detailed metrics tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Faithfulness", "Context Recall", "Answer Correctness", "Answer Relevancy", "Visualizations"])

                with tab1:
                    st.metric("Faithfulness", f"{summary.get('faithfulness', 0):.4f}")
                    st.write("Faithfulness measures how faithful the generated answer is to the provided context.")

                with tab2:
                    st.metric("Context Recall", f"{summary.get('context_recall', 0):.4f}")
                    st.write("Context Recall measures how much of the ground truth information is covered by the retrieved context.")

                with tab3:
                    st.metric("Answer Correctness", f"{summary.get('answer_correctness', 0):.4f}")
                    st.write("Answer Correctness measures the correctness of the generated answer.")

                with tab4:
                    st.metric("Answer Relevancy", f"{summary.get('answer_relevancy', 0):.4f}")
                    st.write("Answer Relevancy measures how relevant the generated answer is to the original question.")

                with tab5:
                    st.header("📊 Evaluation Visualizations")

                    # Check if visualizations were generated
                    if "visualizations" in ragas_result and "error" not in ragas_result["visualizations"]:
                        viz_data = ragas_result["visualizations"]

                        st.success("✅ Visualizations generated successfully!")

                        # Display table
                        if "table" in viz_data:
                            st.subheader("📋 Metrics Table")
                            try:
                                with open(viz_data["table"], 'r', encoding='utf-8') as f:
                                    table_content = f.read()
                                st.markdown(table_content)
                            except Exception as e:
                                st.error(f"Could not load table: {e}")

                        # Display charts
                        chart_files = {k: v for k, v in viz_data.items() if k != "table" and k != "error"}

                        if chart_files:
                            st.subheader("📈 Charts")

                            cols = st.columns(2)  # 2 columns layout

                            for i, (chart_type, chart_path) in enumerate(chart_files.items()):
                                with cols[i % 2]:
                                    st.subheader(f"{chart_type.replace('_', ' ').title()}")
                                    try:
                                        st.image(chart_path, width='stretch')
                                        st.caption(f"File: {chart_path}")
                                    except Exception as e:
                                        st.error(f"Could not load chart {chart_type}: {e}")
                        else:
                            st.info("No chart files were generated.")
                    else:
                        st.warning("⚠️ Visualizations were not generated. They will be created automatically in future evaluations.")

                        # Option to generate visualizations manually
                        if st.button("Generate Visualizations Now", key="generate_viz_now"):
                            st.info("🔄 Generating visualizations...")
                            try:
                                from evaluation.visualizations import RAGMetricsVisualizer
                                visualizer = RAGMetricsVisualizer("data/visualizations")

                                # Prepare data from current results
                                from evaluation.visualizations.utils.data_prep import prepare_metrics_from_ragas_output
                                df = prepare_metrics_from_ragas_output(ragas_result)

                                if not df.empty:
                                    viz_results = visualizer.generate_all_charts(
                                        df,
                                        f"RAG Evaluation - {llm_choice}",
                                        save_charts=True,
                                        show_charts=False
                                    )

                                    if "error" not in viz_results:
                                        st.success("✅ Visualizations generated!")
                                        st.rerun()  # Refresh to show new visualizations
                                    else:
                                        st.error(f"Failed to generate visualizations: {viz_results['error']}")
                                else:
                                    st.error("Could not prepare data for visualizations")

                            except Exception as e:
                                st.error(f"Error generating visualizations: {e}")

                # Configuration summary
                st.subheader("Evaluation Configuration")
                config_data = {
                    'Setting': ['LLM Model', 'Max Questions to Evaluate'],
                    'Value': [llm_choice, str(sample_size) if sample_size > 0 else 'All']
                }
                config_df = pd.DataFrame(config_data)
                st.table(config_df)

                st.success("✅ Ragas evaluation suite completed successfully!")
                try:
                    self.logger.info("Ragas evaluation suite completed successfully")
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Ragas evaluation suite failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                try:
                    self.logger.exception("Ragas evaluation suite failed: %s", e)
                except Exception:
                    pass

    def _render_score_distribution(self, title, results, distribution, score_keys):
        """Render score distribution chart."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader(title)

        # Distribution chart (Matplotlib/Seaborn)
        if distribution:
            dist_df = pd.DataFrame(list(distribution.items()), columns=['Range', 'Count'])
            try:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x='Range', y='Count', data=dist_df, palette='viridis', ax=ax)
                ax.set_xlabel('Range')
                ax.set_ylabel('Count')
                ax.set_title(f'{title} Distribution')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Could not render distribution chart: {e}")

        # Sample results table
        if results:
            sample_results = results[:5]  # Show first 5 results
            if sample_results:
                st.write("Sample Results:")
                sample_data = []
                for result in sample_results:
                    row = {'Question': result.get('question', '')[:50] + '...' if len(result.get('question', '')) > 50 else result.get('question', '')}
                    for key in score_keys:
                        if key in result:
                            row[key.title().replace('_', ' ')] = f"{result[key]:.4f}"
                    sample_data.append(row)
                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df, width='stretch')