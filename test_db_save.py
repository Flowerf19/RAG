from evaluation.backend_dashboard.api import BackendDashboard
import sqlite3

# Test database saving
bd = BackendDashboard()
result = bd.evaluate_ground_truth_with_ragas(llm_provider='ollama', limit=3, save_to_db=True)
print('Database save test completed')

# Check database
conn = sqlite3.connect('data/evaluation.db')
cursor = conn.cursor()

# Count Ragas records
cursor.execute('SELECT COUNT(*) FROM metrics WHERE metadata LIKE "%ragas_ground_truth%"')
count = cursor.fetchone()[0]
print(f'Found {count} Ragas evaluation records in database')

# Get recent records
cursor.execute('SELECT faithfulness, relevance, answer_correctness, recall FROM metrics WHERE metadata LIKE "%ragas_ground_truth%" ORDER BY timestamp DESC LIMIT 3')
rows = cursor.fetchall()
print('Recent records:')
for row in rows:
    print(f'  Faithfulness: {row[0]}, Relevance: {row[1]}, Answer_Correctness: {row[2]}, Recall: {row[3]}')

conn.close()
print('Test completed successfully!')