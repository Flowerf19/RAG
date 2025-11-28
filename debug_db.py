import sqlite3

conn = sqlite3.connect('data/metrics.db')
cursor = conn.cursor()

# Check recent metrics entries
cursor.execute('SELECT id, timestamp, metadata FROM metrics ORDER BY timestamp DESC LIMIT 5')
rows = cursor.fetchall()
print('Recent metrics entries:')
for row in rows:
    metadata = row[2] if row[2] else 'None'
    print(f'  ID: {row[0]}, Timestamp: {row[1]}, Metadata: {metadata[:100]}...')

# Count all metrics
cursor.execute('SELECT COUNT(*) FROM metrics')
count = cursor.fetchone()[0]
print(f'Total metrics records: {count}')

conn.close()