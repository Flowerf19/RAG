import streamlit as st
import pandas as pd
import altair as alt
from evaluation.backend_dashboard.api import BackendDashboard


class PerformanceChartsComponent:
	"""Consolidated performance charts using model comparison data.

	Produces:
	- Latency horizontal bar per model
	- Accuracy vs Recall scatter
	- Error rate horizontal bar per model
	"""

	def __init__(self, backend: BackendDashboard):
		self.backend = backend

	def display(self):
		st.header("Performance Charts")

		data = self.backend.get_model_comparison_data()
		# data is expected to contain lists for 'embedding', 'llm', 'reranking'
		combined = []
		if isinstance(data, dict):
			for k in ['embedding', 'llm', 'reranking']:
				items = data.get(k) or []
				for it in items:
					it_copy = dict(it)
					it_copy['component_type'] = k
					combined.append(it_copy)
		elif isinstance(data, list):
			combined = data

		if not combined:
			st.info("No model performance data available")
			return

		df = pd.DataFrame(combined)

		# Ensure required numeric columns
		for col in ['latency', 'accuracy', 'recall', 'error_rate']:
			if col not in df.columns:
				df[col] = pd.NA

		# Latency per model (horizontal bar sorted descending)
		st.subheader("Latency per Model")
		if 'model' in df.columns:
			lat_df = df[['model', 'latency']].dropna()
			if not lat_df.empty:
				lat_median = lat_df.groupby('model', as_index=False).median()
				lat_chart = (
					alt.Chart(lat_median)
					.mark_bar()
					.encode(
						x=alt.X('latency:Q', title='Median Latency (s)'),
						y=alt.Y('model:N', sort=alt.SortField('latency', order='descending'), title='Model'),
						tooltip=['model', 'latency']
					)
					.properties(height=400)
				)
				st.altair_chart(lat_chart, use_container_width=True)
			else:
				st.info("No latency data available")
		else:
			st.info("Model name column not present")

		# Accuracy vs Recall scatter
		st.subheader("Accuracy vs Recall")
		if 'model' in df.columns:
			ar_df = df[['model', 'accuracy', 'recall', 'latency', 'error_rate']].fillna(0)
			ar_chart = (
				alt.Chart(ar_df)
				.mark_circle()
				.encode(
					x=alt.X('accuracy:Q', title='Accuracy'),
					y=alt.Y('recall:Q', title='Recall'),
					size=alt.Size('latency:Q', title='Latency', scale=alt.Scale(range=[20, 400])),
					color=alt.Color('error_rate:Q', title='Error Rate', scale=alt.Scale(scheme='redblue')),
					tooltip=['model', 'accuracy', 'recall', 'latency', 'error_rate']
				)
				.interactive()
				.properties(height=420)
			)
			st.altair_chart(ar_chart, use_container_width=True)
		else:
			st.info("No accuracy/recall data available")

		# Error rate per model (horizontal)
		st.subheader("Error Rate by Model")
		if 'model' in df.columns and 'error_rate' in df.columns:
			er_df = df[['model', 'error_rate']].dropna()
			if not er_df.empty:
				er_med = er_df.groupby('model', as_index=False).median()
				er_chart = (
					alt.Chart(er_med)
					.mark_bar()
					.encode(
						x=alt.X('error_rate:Q', title='Error Rate'),
						y=alt.Y('model:N', sort=alt.SortField('error_rate', order='descending'), title='Model'),
						color=alt.Color('error_rate:Q', scale=alt.Scale(scheme='redblue')), 
						tooltip=['model', 'error_rate']
					)
					.properties(height=400)
				)
				st.altair_chart(er_chart, use_container_width=True)
			else:
				st.info("No error rate data available")
		else:
			st.info("Error rate data not available")

