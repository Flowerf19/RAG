import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard


class PerformanceChartsComponent:
	"""Consolidated performance charts using model comparison data.

	Produces:
	- Latency bar chart per model
	- Accuracy vs Recall scatter/bar chart
	- Error rate bar chart per model
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
					# ensure model type label
					it_copy = dict(it)
					it_copy['component_type'] = k
					combined.append(it_copy)
		elif isinstance(data, list):
			combined = data

		if not combined:
			st.info("No model performance data available")
			return

		df = pd.DataFrame(combined)

		# Normalize columns we expect
		for col in ['latency', 'accuracy', 'recall', 'error_rate']:
			if col not in df.columns:
				df[col] = pd.NA

		# Latency per model (bar)
		st.subheader("Latency per Model")
		if 'model' in df.columns:
			lat_df = df[['model', 'latency']].dropna()
			if not lat_df.empty:
				lat_df = lat_df.groupby('model').median().sort_values('latency', ascending=False)
				st.bar_chart(lat_df)
			else:
				st.info("No latency data available")
		else:
			st.info("Model name column not present")

		# Accuracy vs Recall (comparison)
		st.subheader("Accuracy and Recall by Model")
		if 'model' in df.columns:
			ar_df = df[['model', 'accuracy', 'recall']].fillna(0).set_index('model')
			if not ar_df.empty:
				st.bar_chart(ar_df)
			else:
				st.info("No accuracy/recall data available")

		# Error rate per model
		st.subheader("Error Rate by Model")
		if 'model' in df.columns and 'error_rate' in df.columns:
			er_df = df[['model', 'error_rate']].dropna().set_index('model').sort_values('error_rate', ascending=False)
			if not er_df.empty:
				st.bar_chart(er_df)
			else:
				st.info("No error rate data available")
		else:
			st.info("Error rate data not available")

