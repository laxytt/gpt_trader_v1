# GPT Flow Visualization Guide

## Overview

The GPT Flow Visualization dashboard provides real-time monitoring and visualization of the Trading Council's GPT API interactions. This feature allows you to:

- Monitor all GPT requests in real-time
- Visualize the flow of information through the Trading Council
- Track token usage and costs
- Analyze agent performance
- Review council decisions and debates

## Features

### 1. Flow Overview

The Flow Overview provides three main views:

#### System Architecture
- Visual representation of the Trading Council structure
- Shows how data flows from market inputs through agents to final decisions
- Interactive Sankey diagram showing information flow

#### Live Flow
- Real-time visualization of active and recent requests
- Timeline view of request processing
- Shows request duration and status

#### Request Status
- Detailed table of recent requests
- View request/response pairs
- Monitor success rates and errors

### 2. Request Timeline

- Gantt chart visualization of all requests
- Grouped by agent type
- Shows processing duration for each request
- Color-coded by agent for easy identification

### 3. Agent Performance

Track performance metrics for each agent:
- Request count
- Average tokens per request
- Total cost by agent
- Cost efficiency (tokens per dollar)

### 4. Token Usage

Monitor token consumption:
- Token usage over time
- Prompt vs completion token breakdown
- Distribution by agent
- Completion/prompt ratio tracking

### 5. Council Decisions

Review Trading Council decisions:
- Visual flow of agent votes
- Debate progression
- Consensus building process
- Individual agent analyses

### 6. Cost Analysis

Track and project API costs:
- Cumulative cost over time
- Cost breakdown by agent
- Hourly/daily/weekly/monthly projections
- Cost per request metrics

## Implementation Details

### Request Monitoring

The system uses a `MonitoredGPTClient` that extends the base GPT client with monitoring capabilities:

```python
from core.infrastructure.gpt.monitored_client import MonitoredGPTClient

# Create monitored client
gpt_client = MonitoredGPTClient(gpt_settings)

# Add monitoring function
def monitor_callback(event_type, request_data):
    if event_type == 'request_start':
        # Log request start
    elif event_type == 'request_complete':
        # Log completion with response
    elif event_type == 'request_failed':
        # Log failure

gpt_client.add_monitor(monitor_callback)
```

### Data Flow Tracking

Each request is tracked with:
- Unique request ID
- Agent type
- Timestamp
- Prompt content (truncated for display)
- Response content (truncated for display)
- Token usage breakdown
- Estimated cost
- Processing duration

### Visualization Components

1. **Sankey Diagrams**: Show information flow through the council
2. **Timeline Charts**: Display request processing over time
3. **Pie Charts**: Break down costs and token usage
4. **Bar Charts**: Compare agent performance
5. **Line Charts**: Track metrics over time

## Usage

### Accessing the Dashboard

1. Start the comprehensive trading dashboard:
   ```bash
   python scripts/comprehensive_trading_dashboard.py
   ```

2. Navigate to "GPT Flow Visualization" in the sidebar

3. Select a view mode from the dropdown

### Simulating Council Sessions

For testing and demonstration:

1. Click the "🎭 Simulate Council Session" button
2. This will generate a mock council session with all agents
3. Watch the requests flow through the system in real-time

### Real-Time Monitoring

When integrated with the live trading system:

1. The dashboard automatically tracks all GPT requests
2. Use auto-refresh to see updates every 5 seconds
3. Monitor active requests in the Live Flow view

## Integration with Trading System

To enable monitoring in your trading system:

1. Replace `GPTClient` with `MonitoredGPTClient` in your dependency injection:

```python
from core.infrastructure.gpt.monitored_client import MonitoredGPTClient

# In your dependency container
self.gpt_client = MonitoredGPTClient(settings.gpt)
```

2. The monitoring will automatically track all requests through the Trading Council

3. Access the dashboard to view real-time data

## Performance Considerations

- Request history is limited to 1000 entries to prevent memory issues
- Older requests are automatically pruned
- Dashboard updates can be set to manual refresh for better performance
- Large council sessions may take 10-30 seconds to complete

## Troubleshooting

### No Data Showing

- Ensure the trading system is using `MonitoredGPTClient`
- Check that requests are being made to the GPT API
- Try simulating a council session to verify the dashboard works

### Performance Issues

- Disable auto-refresh if the dashboard is slow
- Reduce the number of displayed requests
- Clear request history if needed

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Future Enhancements

Planned improvements:
- Persistent storage of request history
- Advanced filtering and search
- Export capabilities for analysis
- Alerting on high costs or errors
- Integration with model performance metrics