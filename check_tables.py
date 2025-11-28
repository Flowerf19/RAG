import sqlite3

conn = sqlite3.connect('data/evaluation.db')
cursor = conn.cursor()

# Check what tables exist
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

conn.close()