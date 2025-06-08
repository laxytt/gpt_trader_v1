import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Connect to the database
db_path = Path("data/gpt_requests.db")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get recent requests
cursor.execute("""
    SELECT timestamp, agent_type, total_tokens, duration_ms, cost
    FROM gpt_requests 
    ORDER BY timestamp DESC 
    LIMIT 20
""")

results = cursor.fetchall()

print(f"Found {len(results)} recent requests:\n")
print(f"{'Timestamp':<30} {'Agent':<25} {'Tokens':<10} {'Duration':<10} {'Cost':<10}")
print("-" * 90)

for row in results:
    timestamp_str, agent, tokens, duration, cost = row
    # Parse timestamp
    try:
        # Try parsing ISO format
        if 'T' in timestamp_str:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        # Format for display
        formatted_ts = ts.strftime('%Y-%m-%d %H:%M:%S %Z')
    except:
        formatted_ts = timestamp_str[:30]
    
    agent = agent or 'N/A'
    tokens = tokens or 0
    duration_ms = duration or 0
    duration_s = duration_ms / 1000 if duration_ms else 0
    cost = cost or 0
    
    print(f"{formatted_ts:<30} {agent:<25} {tokens:<10} {duration_s:<10.2f} ${cost:<10.4f}")

# Get count by hour
print("\n\nRequests by hour:")
cursor.execute("""
    SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour, COUNT(*) as count
    FROM gpt_requests
    WHERE timestamp > datetime('now', '-24 hours')
    GROUP BY hour
    ORDER BY hour DESC
    LIMIT 10
""")

hour_results = cursor.fetchall()
for hour, count in hour_results:
    print(f"{hour}: {count} requests")

conn.close()