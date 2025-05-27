"""
GPT reflection generator for analyzing completed trades.
Provides post-trade analysis and learning insights for strategy improvement.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from core.infrastructure.gpt.client import GPTClient
from core.domain.models import Trade, MarketData, TradeResult
from core.domain.exceptions import (
    ReflectionGenerationError, ErrorContext
)


logger = logging.getLogger(__name__)


class GPTReflectionGenerator:
    """
    Generates trade reflection analysis using GPT for completed trades.
    Provides insights for strategy improvement and learning.
    """
    
    def __init__(self, gpt_client: GPTClient, prompts_dir: str = "config/prompts"):
        self.gpt_client = gpt_client
        
        # Load reflection prompt template
        self.reflection_prompt = self._load_reflection_prompt(prompts_dir)
    
    def _load_reflection_prompt(self, prompts_dir: str) -> str:
        """Load the reflection prompt template from file"""
        try:
            prompt_path = Path(prompts_dir) / "reflection_prompt.txt"
            
            # If specific reflection prompt doesn't exist, create a default one
            if not prompt_path.exists():
                return self._get_default_reflection_prompt()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error loading reflection prompt: {e}, using default")
            return self._get_default_reflection_prompt()
    
    def _get_default_reflection_prompt(self) -> str:
        """Get default reflection prompt if file not available"""
        return """
You are an expert trading coach analyzing a completed trade using VSA (Volume Spread Analysis) and multi-timeframe technical analysis principles.

Your task is to provide constructive feedback on the trade execution and outcome, focusing on:

1. **Signal Quality Assessment**
   - Was the entry signal valid according to VSA principles?
   - Did the H4 background support the H1 entry?
   - Were momentum and volume confirmation adequate?

2. **Execution Analysis**
   - Was the entry timing optimal?
   - Were stop-loss and take-profit levels appropriate?
   - Did the trade follow the planned strategy?

3. **Market Context Evaluation**  
   - How did news events affect the trade?
   - Was market volatility conducive to the strategy?
   - Did session timing play a role in the outcome?

4. **Learning Points**
   - What worked well in this trade?
   - What could be improved for similar future setups?
   - Are there pattern recognition insights to apply?

Provide your analysis in a concise, actionable paragraph focusing on the most important learning points. Be specific about VSA patterns, market structure, and risk management observations.
"""
    
    async def generate_reflection(
        self,
        trade: Trade,
        h1_data: Optional[MarketData] = None,
        h4_data: Optional[MarketData] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate reflection analysis for a completed trade.
        
        Args:
            trade: Completed trade to analyze
            h1_data: Optional H1 market data around trade time
            h4_data: Optional H4 market data around trade time  
            additional_context: Optional additional context data
            
        Returns:
            Reflection analysis as string
            
        Raises:
            ReflectionGenerationError: If reflection generation fails
        """
        if trade.result is None:
            raise ReflectionGenerationError("Cannot generate reflection for trade without result")
        
        with ErrorContext("Trade reflection generation", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            ctx.add_detail("trade_result", trade.result.value if trade.result else None)
            ctx.add_detail("duration_minutes", trade.duration_minutes)
            
            # Prepare reflection data
            reflection_data = self._prepare_reflection_data(
                trade, h1_data, h4_data, additional_context
            )
            
            # Build GPT messages
            messages = self._build_reflection_messages(reflection_data)
            
            # Call GPT API
            gpt_response = await self.gpt_client.chat_completion(
                messages=messages,
                temperature=0.3,  # Slightly higher for more creative analysis
                max_tokens=800    # Sufficient for detailed reflection
            )
            
            reflection_text = gpt_response['content'].strip()
            
            if not reflection_text:
                raise ReflectionGenerationError("Empty reflection response from GPT")
            
            logger.info(f"Reflection generated for trade {trade.id} ({trade.symbol})")
            
            return reflection_text
    
    def _prepare_reflection_data(
        self,
        trade: Trade,
        h1_data: Optional[MarketData],
        h4_data: Optional[MarketData],
        additional_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for reflection analysis"""
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics(trade)
        
        # Prepare trade summary
        trade_summary = {
            'symbol': trade.symbol,
            'side': trade.side.value,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'result': trade.result.value if trade.result else None,
            'duration_minutes': trade.duration_minutes,
            'pnl': trade.current_pnl,
            'risk_reward_planned': trade.risk_reward_ratio,
            'risk_reward_actual': trade_metrics.get('actual_rr'),
            'max_drawdown_pips': trade.max_drawdown_pips,
            'timestamp_entry': trade.timestamp.isoformat(),
            'timestamp_exit': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None
        }
        
        # Add original signal context if available
        signal_context = {}
        if trade.original_signal:
            signal_context = {
                'signal_reason': trade.original_signal.reason,
                'risk_class': trade.original_signal.risk_class.value,
                'market_context': trade.original_signal.market_context
            }
        
        # Add management history
        management_summary = self._summarize_management_actions(trade.management_history)
        
        reflection_data = {
            'trade_summary': trade_summary,
            'signal_context': signal_context,
            'management_actions': management_summary,
            'trade_metrics': trade_metrics,
            'market_data_h1': self._summarize_market_data(h1_data) if h1_data else None,
            'market_data_h4': self._summarize_market_data(h4_data) if h4_data else None,
            'additional_context': additional_context or {}
        }
        
        return reflection_data
    
    def _calculate_trade_metrics(self, trade: Trade) -> Dict[str, Any]:
        """Calculate additional trade performance metrics"""
        metrics = {}
        
        if trade.exit_price and trade.entry_price and trade.stop_loss:
            # Calculate actual risk/reward
            if trade.side.value == "BUY":
                risk_pips = abs(trade.entry_price - trade.stop_loss)
                reward_pips = abs(trade.exit_price - trade.entry_price)
            else:
                risk_pips = abs(trade.stop_loss - trade.entry_price)  
                reward_pips = abs(trade.entry_price - trade.exit_price)
            
            if risk_pips > 0:
                metrics['actual_rr'] = round(reward_pips / risk_pips, 2)
            
            metrics['risk_pips'] = round(risk_pips * 10000, 1)  # Convert to pips
            metrics['reward_pips'] = round(reward_pips * 10000, 1)
        
        # Trade efficiency (how much of available move was captured)
        if trade.take_profit and trade.exit_price:
            if trade.side.value == "BUY":
                potential_reward = abs(trade.take_profit - trade.entry_price)
                actual_reward = abs(trade.exit_price - trade.entry_price)
            else:
                potential_reward = abs(trade.entry_price - trade.take_profit)
                actual_reward = abs(trade.entry_price - trade.exit_price)
            
            if potential_reward > 0:
                metrics['efficiency_percent'] = round((actual_reward / potential_reward) * 100, 1)
        
        # Drawdown analysis
        metrics['max_drawdown_pips'] = trade.max_drawdown_pips
        
        return metrics
    
    def _summarize_management_actions(self, management_history: list) -> Dict[str, Any]:
        """Summarize trade management actions"""
        if not management_history:
            return {'total_actions': 0, 'types': []}
        
        action_types = [action.get('action', 'unknown') for action in management_history]
        action_counts = {}
        for action_type in action_types:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            'total_actions': len(management_history),
            'action_counts': action_counts,
            'timeline': [
                {
                    'timestamp': action.get('timestamp'),
                    'action': action.get('action'),
                    'details': action.get('details', {})
                }
                for action in management_history[-3:]  # Last 3 actions only
            ]
        }
    
    def _summarize_market_data(self, market_data: MarketData) -> Dict[str, Any]:
        """Summarize market data for reflection context"""
        if not market_data.candles:
            return {}
        
        latest_candle = market_data.latest_candle
        if not latest_candle:
            return {}
        
        return {
            'timeframe': market_data.timeframe,
            'latest_candle': {
                'timestamp': latest_candle.timestamp.isoformat(),
                'close': latest_candle.close,
                'volume': latest_candle.volume,
                'ema50': latest_candle.ema50,
                'ema200': latest_candle.ema200,
                'rsi14': latest_candle.rsi14,
                'atr14': latest_candle.atr14
            },
            'total_candles': len(market_data.candles)
        }
    
    def _build_reflection_messages(self, reflection_data: Dict[str, Any]) -> list:
        """Build messages for GPT reflection analysis"""
        
        # Format trade summary for analysis
        trade_summary = reflection_data['trade_summary']
        signal_context = reflection_data['signal_context']
        
        trade_description = f"""
Trade Analysis Request:

**Trade Summary:**
- Symbol: {trade_summary['symbol']}
- Direction: {trade_summary['side']}
- Entry: {trade_summary['entry_price']}
- Exit: {trade_summary['exit_price']}
- Stop Loss: {trade_summary['stop_loss']}  
- Take Profit: {trade_summary['take_profit']}
- Result: {trade_summary['result']}
- Duration: {trade_summary['duration_minutes']:.1f} minutes
- P&L: {trade_summary['pnl']:.2f}

**Performance Metrics:**
- Planned R/R: {trade_summary['risk_reward_planned']}
- Actual R/R: {reflection_data['trade_metrics'].get('actual_rr', 'N/A')}
- Max Drawdown: {trade_summary['max_drawdown_pips']:.1f} pips
- Efficiency: {reflection_data['trade_metrics'].get('efficiency_percent', 'N/A')}%

**Original Signal Context:**
- Reason: {signal_context.get('signal_reason', 'N/A')}
- Risk Class: {signal_context.get('risk_class', 'N/A')}

**Management Actions:**
- Total Actions: {reflection_data['management_actions']['total_actions']}
- Action Types: {reflection_data['management_actions'].get('action_counts', {})}
"""
        
        # Add market data context if available
        if reflection_data.get('market_data_h1'):
            h1_data = reflection_data['market_data_h1']
            trade_description += f"""
**H1 Market Context:**
- Latest Close: {h1_data['latest_candle']['close']}
- RSI: {h1_data['latest_candle']['rsi14']}
- ATR: {h1_data['latest_candle']['atr14']}
"""
        
        messages = [
            {"role": "system", "content": self.reflection_prompt},
            {"role": "user", "content": trade_description}
        ]
        
        return messages
    
    def generate_batch_reflection(
        self,
        trades: list[Trade],
        summary_focus: str = "overall_patterns"
    ) -> str:
        """
        Generate reflection on a batch of trades for pattern analysis.
        
        Args:
            trades: List of completed trades
            summary_focus: Focus area for batch analysis
            
        Returns:
            Batch reflection analysis
        """
        # This could be implemented for analyzing multiple trades together
        # to identify broader patterns and strategic improvements
        raise NotImplementedError("Batch reflection not yet implemented")


# Export main class
__all__ = ['GPTReflectionGenerator']