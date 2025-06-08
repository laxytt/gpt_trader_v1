"""
Configure Trading Council agents to follow specific strategies
"""

# You can modify agent prompts to include your VSA rules
VSA_STRATEGY_RULES = """
IMPORTANT: Apply these VSA (Volume Spread Analysis) rules in your analysis:

1. Volume Analysis:
   - Rising price + rising volume = Strength
   - Rising price + falling volume = Weakness (no demand)
   - Falling price + rising volume = Weakness continues
   - Falling price + falling volume = Strength (no supply)

2. Key VSA Signals to Look For:
   
   BULLISH SIGNALS:
   - Stopping Volume: High volume after decline, narrow spread
   - Shake Out: Break below support but close near high
   - No Supply: Low volume test of lows
   - Spring: False breakdown with rapid recovery
   
   BEARISH SIGNALS:
   - Buying Climax: Very high volume, wide spread, close mid-bar
   - Upthrust: High volume, long upper shadow, close near low
   - No Demand: Low volume on attempted rallies
   - Distribution: High volume but no upward progress

3. Background Analysis:
   - Always check higher timeframe first
   - Identify last major accumulation/distribution
   - Note position within trading range

4. Entry Rules:
   - Wait for test after accumulation signal
   - Enter on successful low-volume test
   - Stop loss below shake out/spring low
"""

# To use: Add this to each agent's prompt or create specialized VSA agents