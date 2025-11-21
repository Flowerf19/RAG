import sqlite3

conn = sqlite3.connect('data/metrics.db')
cursor = conn.cursor()

# Check what tables exist
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

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