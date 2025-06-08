"""
Fix for GPT Flow Dashboard Issues
"""

# Fix 1: Add missing method to SignalRepository
def add_get_all_signals_method():
    """Add the missing get_all_signals method to SignalRepository"""
    
    code_to_add = '''
    def get_all_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all signals for dashboard display"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    symbol,
                    signal as direction,
                    confidence,
                    created_at,
                    entry_price,
                    stop_loss,
                    take_profit,
                    market_context
                FROM signals
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            signals = []
            for row in cursor.fetchall():
                signal = dict(row)
                # Parse market context if it's JSON
                if signal.get('market_context'):
                    try:
                        import json
                        signal['analysis'] = json.loads(signal['market_context'])
                    except:
                        signal['analysis'] = {}
                signals.append(signal)
            
            return signals
'''
    
    print("Add this method to SignalRepository in core/infrastructure/database/repositories.py:")
    print(code_to_add)

# Fix 2: Update dashboard to handle timezone properly
def fix_timestamp_display():
    """Fix timestamp display in dashboard"""
    
    dashboard_fix = '''
# In gpt_flow_dashboard.py, update the show_request_payloads method:

# Add this import at the top
from datetime import timezone

# In the request display section, update timestamp formatting:
for req in requests:
    # Parse timestamp
    timestamp_str = req.get('timestamp', '')
    if timestamp_str:
        try:
            # Parse ISO timestamp
            if 'T' in timestamp_str:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Convert to local time for display
            local_ts = ts.astimezone()
            time_str = local_ts.strftime('%H:%M:%S')
        except:
            time_str = timestamp_str.split('T')[1][:8] if 'T' in timestamp_str else timestamp_str[11:19]
'''
    
    print("\n\nFix for timestamp display:")
    print(dashboard_fix)

# Fix 3: Ensure request logger is properly filtering by time
def fix_time_filtering():
    """Fix time filtering in request logger"""
    
    filter_fix = '''
# In request_logger.py, update get_recent_requests to filter by time:

def get_recent_requests(
    self, 
    limit: int = 100,
    hours_back: int = 24,
    agent_type: Optional[str] = None,
    symbol: Optional[str] = None,
    include_errors: bool = True
) -> List[Dict[str, Any]]:
    """Get recent requests for dashboard"""
    try:
        with self._get_connection() as conn:
            query = """
                SELECT * FROM gpt_requests
                WHERE timestamp > datetime('now', '-{} hours')
            """.format(hours_back)
            
            params = []
            
            if agent_type:
                query += " AND agent_type = ?"
                params.append(agent_type)
                
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            if not include_errors:
                query += " AND error IS NULL"
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            requests = []
            for row in rows:
                req = dict(row)
                req['messages'] = json.loads(req['messages'])
                requests.append(req)
                
            return requests
'''
    
    print("\n\nFix for time filtering:")
    print(filter_fix)

if __name__ == "__main__":
    print("Dashboard Issue Fixes")
    print("=" * 50)
    
    add_get_all_signals_method()
    fix_timestamp_display()
    fix_time_filtering()
    
    print("\n\nQuick Fix Instructions:")
    print("1. Add the get_all_signals method to SignalRepository")
    print("2. Update timestamp handling in the dashboard")
    print("3. Add hours_back parameter to request filtering")
    print("4. Restart the dashboard after making changes")