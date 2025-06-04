# Trading Council Multi-Agent System Guide

## Overview

The Trading Council is a sophisticated multi-agent decision-making system that replaces the traditional single GPT-based signal generation. It implements a council of 7 specialized AI agents that analyze markets from different perspectives, debate their views, and reach consensus decisions.

## Architecture

### Agent Types

1. **Technical Analyst** - Chart patterns, indicators, support/resistance
2. **Fundamental Analyst** - News impact, economic context
3. **Sentiment Reader** - Market psychology, crowd behavior
4. **Risk Manager** - Capital preservation, position sizing (has veto power)
5. **Momentum Trader** - Trend strength, breakouts
6. **Contrarian Trader** - Reversals, fading extremes
7. **Head Trader** - Synthesizes all views into final decision

### Decision Process

The council follows a structured 4-phase process:

#### Phase 1: Individual Analysis
- All agents (except Head Trader) analyze market data in parallel
- Each produces an independent recommendation with confidence

#### Phase 2: Three-Round Debate
- **Round 1**: Opening statements
- **Round 2**: Counter-arguments and challenges
- **Round 3**: Final positions
- Head Trader moderates the discussion

#### Phase 3: Synthesis
- Head Trader weighs all arguments
- Considers agent expertise and track records
- Makes final trading decision

#### Phase 4: Confidence Scoring
- LLM confidence based on consensus (70% weight)
- ML model confidence as reference (30% weight)
- Final confidence must exceed threshold to trade

## Configuration

Add these settings to your `.env` file:

```env
# Trading Council Configuration
TRADING_USE_COUNCIL=true
TRADING_COUNCIL_MIN_CONFIDENCE=75.0
TRADING_COUNCIL_LLM_WEIGHT=0.7
TRADING_COUNCIL_ML_WEIGHT=0.3
TRADING_COUNCIL_DEBATE_ROUNDS=3
```

### Settings Explained

- `USE_COUNCIL`: Enable/disable the multi-agent system
- `COUNCIL_MIN_CONFIDENCE`: Minimum confidence (0-100) required to take a trade
- `COUNCIL_LLM_WEIGHT`: Weight for agent consensus in final confidence (0-1)
- `COUNCIL_ML_WEIGHT`: Weight for ML model predictions (0-1)
- `COUNCIL_DEBATE_ROUNDS`: Number of debate rounds (1-5, default 3)

## How It Works

### 1. Market Data Gathering
```python
# The council receives:
- H1 timeframe data (entry decisions)
- H4 timeframe data (background context)
- Recent news events
- ML model predictions (if available)
```

### 2. Agent Personalities

Each agent has a unique personality and approach:

- **Technical Analyst**: "Methodical and detail-oriented"
- **Fundamental Analyst**: "Strategic big-picture thinker"
- **Sentiment Reader**: "Intuitive market psychologist"
- **Risk Manager**: "Conservative capital protector"
- **Momentum Trader**: "Aggressive trend follower"
- **Contrarian Trader**: "Skeptical fade trader"
- **Head Trader**: "Balanced decisive leader"

### 3. Debate Dynamics

Agents can:
- Challenge each other's views
- Change positions based on arguments
- Form coalitions or dissent
- Provide specific execution advice

### 4. Risk Management

The Risk Manager has special privileges:
- Can veto high-risk trades
- Sets position sizes based on account risk
- Influences confidence when risk is high

## Testing the Council

Use the provided test script:

```bash
python test_council.py
```

This will:
1. Connect to MT5
2. Analyze a test symbol (EURUSD)
3. Show the complete council deliberation
4. Display the final decision with metadata

## Monitoring Council Performance

The system tracks:
- Individual agent performance
- Consensus levels over time
- Decision accuracy
- Dissent patterns

Access performance data:
```python
# Get recent council decisions
history = await signal_service.get_council_history(limit=10)

# Get agent performance metrics
performance = await signal_service.get_agent_performance()
```

## Advantages Over Single GPT

1. **Diverse Perspectives**: Multiple viewpoints reduce blind spots
2. **Debate Process**: Arguments are challenged and refined
3. **Specialization**: Each agent excels in their domain
4. **Transparency**: Clear reasoning from each perspective
5. **Risk Awareness**: Dedicated risk management voice
6. **Adaptability**: Agents can evolve their strategies

## Integration with Existing System

The Trading Council seamlessly integrates with:
- **Offline Validation**: Pre-screens market quality
- **News Service**: Provides economic context
- **Memory Service**: Historical trade patterns
- **Chart Generation**: Visual analysis support
- **ML Models**: Reference predictions

## Troubleshooting

### Common Issues

1. **Low Consensus**
   - Normal for unclear markets
   - System will output WAIT signal
   - Check agent reasoning for insights

2. **High Dissent**
   - Indicates market uncertainty
   - Review dissenting views in metadata
   - May signal regime change

3. **Slow Performance**
   - Council analysis takes 30-60 seconds
   - This is normal for thorough analysis
   - Can reduce debate rounds if needed

### Debug Mode

Enable detailed logging:
```python
logging.getLogger('core.agents').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements:
1. Agent learning from outcomes
2. Dynamic agent weights based on performance
3. Specialized agents for different market conditions
4. Real-time debate visualization
5. Custom agent personalities per trading style

## Best Practices

1. **Let the Council Work**: Don't override decisions without good reason
2. **Monitor Consensus**: Low consensus often precedes volatility
3. **Review Dissent**: Dissenting views often contain valuable insights
4. **Track Performance**: Regularly review which agents perform best
5. **Adjust Confidence**: Tune thresholds based on your risk tolerance

The Trading Council represents a significant advancement in algorithmic trading decision-making, bringing the power of collective intelligence to every trade decision.