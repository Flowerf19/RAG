from evaluation.backend_dashboard.api import BackendDashboard

backend = BackendDashboard()
rows = backend.get_ground_truth_list(limit=5)
print(f'Found {len(rows)} ground truth rows')
for row in rows:
    print(f'ID {row["id"]}: {row["question"][:50]}...')