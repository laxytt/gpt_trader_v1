"""
Risk Manager Agent
Specializes in position sizing, risk assessment, and capital preservation
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class RiskManager(TradingAgent):
    """The Guardian - protects capital and manages risk"""
    
    def __init__(self, gpt_client: GPTClient, account_balance: float = 10000, risk_per_trade: float = 0.02):
        super().__init__(
            agent_type=AgentType.RISK_MANAGER,
            personality="conservative and protective risk guardian",
            specialty="position sizing, stop placement, risk/reward ratios, and capital preservation"
        )
        self.gpt_client = gpt_client
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 2% default
        self.max_risk_per_trade = risk_per_trade * account_balance
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze from professional risk management perspective"""
        
        # Check if we're in position trading mode
        is_position_mode = self._is_position_trading_mode(market_data)
        
        if is_position_mode:
            return self._analyze_position_risk(market_data, news_context, ml_context)
        
        # Standard risk analysis for scalping/day trading
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        d1_data = market_data.get('d1')
        
        if not h1_data:
            raise ValueError("H1 data required for risk analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            import numpy as np
            analyzer = ProfessionalMarketAnalyzer()
            
            # Need 250+ candles for proper risk metrics
            h1_candles = h1_data.candles[-250:] if len(h1_data.candles) >= 250 else h1_data.candles
            h4_candles = h4_data.candles[-100:] if h4_data and len(h4_data.candles) >= 100 else []
            
            # Professional volatility metrics
            realized_vol = analyzer.calculate_realized_volatility(h1_candles, 20)
            regime = analyzer.calculate_market_regime(h1_candles)
            
            use_enhanced = True
        except:
            use_enhanced = False
            realized_vol = 0.1
            regime = {'regime': 'unknown'}
        
        current_price = h1_data.latest_candle.close
        atr = h1_data.latest_candle.atr14
        spread = h1_data.latest_candle.spread or 0.0001
        
        # Professional risk calculations
        candles = h1_data.candles
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = self._calculate_var(candles, 0.95)
        cvar_95 = self._calculate_cvar(candles, 0.95)
        
        # Maximum Adverse Excursion
        mae = self._calculate_mae(candles)
        
        # Volatility analysis
        vol_percentile = self._calculate_volatility_percentile(candles, atr)
        vol_regime = self._classify_volatility_regime(realized_vol)
        
        # Correlation risk (simplified)
        correlation_risk = self._assess_correlation_risk(h1_data.symbol)
        
        # Kelly Criterion position sizing
        win_rate, avg_win, avg_loss = self._calculate_win_loss_stats(candles)
        kelly_fraction = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Technical stop placement analysis
        tech_stops = self._find_technical_stop_levels(candles)
        
        # Spread cost analysis
        spread_cost_pct = (spread / current_price) * 100
        spread_impact = spread / atr if atr > 0 else float('inf')
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} from a professional risk management perspective.

VOLATILITY METRICS (250 bars):
- Current ATR(14): {atr_pips:.1f} pips
- Realized Vol (annualized): {realized_vol:.1%}
- Vol Percentile: {vol_percentile:.0f}th
- Vol Regime: {vol_regime}
- Market Regime: {regime}

RISK METRICS:
- VaR (95%): {var_95:.1f} pips
- CVaR (95%): {cvar_95:.1f} pips
- Max Adverse Excursion: {mae:.1f} pips
- Current Spread: {spread_pips:.1f} pips ({spread_cost:.3f}%)
- Spread/ATR Impact: {spread_impact:.1%}

POSITION SIZING:
- Account Risk Limit: ${max_risk:.2f} ({risk_percent}%)
- Kelly Fraction: {kelly:.1%}
- Win Rate: {win_rate:.1%}
- Avg Win/Loss: {avg_win:.1f}/{avg_loss:.1f} pips

TECHNICAL LEVELS:
- Current Price: {price:.5f}
- Natural Stop Levels: {stop_levels}
- Correlation Risk: {corr_risk}

MARKET CONDITIONS:
- 24h Range: {range_24h:.1f} pips
- 5-day Range: {range_5d:.1f} pips
- Price vs 20-day range: {price_position:.1%}

Professional Risk Assessment Required:
1. Position sizing based on volatility regime
2. Stop placement for maximum edge
3. Risk/Reward optimization
4. Portfolio heat assessment
5. Black swan protection
6. Execution risk (slippage, gaps)

Provide:
RISK_LEVEL: [Low/Medium/High/Extreme]
POSITION_SIZE: [0.01-1.0 lots based on Kelly]
STOP_DISTANCE: [optimal pips]
RISK_REWARD: [minimum acceptable ratio]
MAX_LOSS: [dollar amount]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
KEY_RISKS: [prioritized risk factors]
VAR_BREACH: [probability of exceeding loss limit]

Format your response with exact labels above.
""",
            symbol=h1_data.symbol,
            price=current_price,
            atr_pips=atr * 10000,
            realized_vol=realized_vol,
            vol_percentile=vol_percentile,
            vol_regime=vol_regime,
            regime=regime.get('regime', 'unknown'),
            var_95=var_95,
            cvar_95=cvar_95,
            mae=mae,
            spread_pips=spread * 10000,
            spread_cost=spread_cost_pct,
            spread_impact=spread_impact,
            max_risk=self.max_risk_per_trade,
            risk_percent=self.risk_per_trade * 100,
            kelly=kelly_fraction,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            stop_levels=", ".join([f"{level:.5f}" for level in tech_stops[:3]]),
            corr_risk=correlation_risk,
            range_24h=self._calculate_range(candles[-24:]) * 10000 if len(candles) >= 24 else 0,
            range_5d=self._calculate_range(candles[-120:]) * 10000 if len(candles) >= 120 else 0,
            price_position=self._calculate_price_position(candles)
        )
        
        # Risk Manager uses GPT-4 for its veto power
        from config.settings import get_settings
        settings = get_settings()
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.2,  # Very low temperature for conservative risk assessment
            model_override=settings.gpt.risk_manager_model,
            agent_type=self.agent_type.value,
            symbol=h1_data.symbol
        )
        
        return self._parse_analysis(response, current_price)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate focusing on risk"""
        
        my_position = self.analysis_history[-1]
        
        # Risk manager has veto power on high-risk trades
        high_risk_agents = [
            a for a in other_analyses 
            if a.agent_type in [AgentType.MOMENTUM_TRADER, AgentType.SENTIMENT_READER]
            and a.recommendation != SignalType.WAIT
            and a.confidence > 80
        ]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2 and high_risk_agents:
            statement = self._address_aggressive_traders(high_risk_agents[0])
        else:
            statement = self._generate_closing_statement(other_analyses, my_position)
        
        # Risk manager may strengthen position if others are too aggressive
        updated_confidence = my_position.confidence
        
        aggressive_count = sum(1 for a in other_analyses if a.recommendation != SignalType.WAIT and a.confidence > 70)
        if aggressive_count >= 4 and my_position.recommendation == SignalType.WAIT:
            updated_confidence = min(100, my_position.confidence + 20)  # Stronger conviction to wait
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Risk manager rarely changes position
            updated_confidence=updated_confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis with safe parsing"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            raw_parsed = parsed_data['metadata']['raw_parsed']
            
            # Risk manager is conservative - high bar for trades
            risk_level = raw_parsed.get('RISK_LEVEL', 'High').upper()
            
            if 'HIGH' in risk_level:
                recommendation = SignalType.WAIT
                confidence = 90.0  # Very high confidence in waiting when risk is high
            else:
                rec_text = raw_parsed.get('RECOMMENDATION', 'WAIT').upper()
                if 'BUY' in rec_text and 'LOW' in risk_level:
                    recommendation = SignalType.BUY
                elif 'SELL' in rec_text and 'LOW' in risk_level:
                    recommendation = SignalType.SELL
                else:
                    recommendation = SignalType.WAIT
                
                confidence = parsed_data['confidence']
            
            # Extract risk metrics safely
            try:
                stop_distance = float(raw_parsed.get('STOP_DISTANCE', '20').replace('pips', '').strip())
            except:
                stop_distance = 20.0
                
            try:
                risk_reward_str = raw_parsed.get('RISK_REWARD', '1.0')
                risk_reward = float(risk_reward_str.replace(':1', '').strip())
            except:
                risk_reward = 1.0
                
            try:
                position_size = float(raw_parsed.get('POSITION_SIZE', '0.1'))
            except:
                position_size = 0.1
            
            # Calculate stop loss and take profit
            if recommendation == SignalType.BUY:
                stop_loss = current_price - (stop_distance / 10000)
                take_profit = current_price + (stop_distance * risk_reward / 10000)
            elif recommendation == SignalType.SELL:
                stop_loss = current_price + (stop_distance / 10000)
                take_profit = current_price - (stop_distance * risk_reward / 10000)
            else:
                stop_loss = 0
                take_profit = 0
            
            # Build reasoning
            reasoning = []
            reasoning.append(f"Risk level: {risk_level}")
            if raw_parsed.get('RISK_REWARD'):
                reasoning.append(f"Risk/Reward: {raw_parsed['RISK_REWARD']}")
            reasoning.append(f"Position size: {position_size} lots")
            
            concerns = []
            if raw_parsed.get('KEY_RISKS'):
                concerns.append(raw_parsed['KEY_RISKS'])
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning[:3] or ["Risk assessment completed"],
                concerns=concerns,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                metadata={
                    'risk_level': risk_level,  # CRITICAL for veto logic
                    'position_size': position_size,
                    'stop_distance': stop_distance,
                    'max_loss': raw_parsed.get('MAX_LOSS', 'Unknown')
                }
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            # If parsing fails, default to maximum safety
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=95.0,  # Very high confidence in safety
                reasoning=["Risk analysis failed - defaulting to maximum safety"],
                concerns=["Unable to properly assess risk - trade rejected"],
                entry_price=current_price,
                metadata={
                    'risk_level': 'High',  # Default to high risk
                    'error': str(e)
                }
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        risk_level = analysis.metadata.get('risk_level', 'Unknown')
        if analysis.recommendation == SignalType.WAIT:
            return f"Risk assessment shows {risk_level} risk. We should wait for better conditions. Capital preservation is paramount."
        else:
            return f"Risk is {risk_level} and manageable. {analysis.reasoning[1]}. We can proceed with proper position sizing."
    
    def _address_aggressive_traders(self, aggressive_agent: AgentAnalysis) -> str:
        """Address aggressive trading suggestions"""
        return f"The {aggressive_agent.agent_type.value} suggests aggressive entry, but consider the downside. One bad trade can erase multiple winners. Let's wait for optimal risk/reward."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis], my_position: AgentAnalysis) -> str:
        """Generate closing statement"""
        risk_level = my_position.metadata.get('risk_level', 'Unknown')
        
        if my_position.recommendation == SignalType.WAIT:
            return f"With {risk_level} risk identified, protecting capital takes priority. No trade is better than a bad trade."
        else:
            position_size = my_position.metadata.get('position_size', 0.1)
            return f"If we proceed, strict risk management is essential. Maximum {position_size} lots with stops at technical levels."
    
    def _calculate_var(self, candles: List, confidence: float = 0.95) -> float:
        """Calculate Value at Risk in pips"""
        if len(candles) < 20:
            return 20.0  # Default
            
        # Calculate returns
        returns = []
        for i in range(1, len(candles)):
            ret = (candles[i].close - candles[i-1].close) * 10000  # In pips
            returns.append(ret)
        
        if not returns:
            return 20.0
            
        # Sort returns
        returns.sort()
        
        # Find VaR at confidence level
        index = int((1 - confidence) * len(returns))
        return abs(returns[index]) if index < len(returns) else abs(returns[0])
    
    def _calculate_cvar(self, candles: List, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(candles) < 20:
            return 25.0
            
        returns = []
        for i in range(1, len(candles)):
            ret = (candles[i].close - candles[i-1].close) * 10000
            returns.append(ret)
        
        if not returns:
            return 25.0
            
        returns.sort()
        index = int((1 - confidence) * len(returns))
        
        # Average of returns worse than VaR
        tail_returns = returns[:index] if index > 0 else returns[:1]
        return abs(sum(tail_returns) / len(tail_returns)) if tail_returns else 25.0
    
    def _calculate_mae(self, candles: List) -> float:
        """Calculate Maximum Adverse Excursion"""
        if len(candles) < 20:
            return 15.0
            
        mae_values = []
        
        # Simulate entries at each candle
        for i in range(len(candles) - 20):
            entry_price = candles[i].close
            
            # Track maximum adverse move over next 20 candles
            max_adverse = 0
            for j in range(i + 1, min(i + 20, len(candles))):
                # For both long and short
                adverse_long = (entry_price - candles[j].low) * 10000
                adverse_short = (candles[j].high - entry_price) * 10000
                max_adverse = max(max_adverse, adverse_long, adverse_short)
            
            mae_values.append(max_adverse)
        
        # Return 75th percentile MAE
        if mae_values:
            mae_values.sort()
            index = int(0.75 * len(mae_values))
            return mae_values[index]
        return 15.0
    
    def _calculate_volatility_percentile(self, candles: List, current_atr: float) -> float:
        """Calculate where current volatility ranks historically"""
        if len(candles) < 100:
            return 50.0
            
        # Calculate historical ATRs
        atr_values = []
        for i in range(14, len(candles)):
            # Simplified ATR calculation
            tr_sum = 0
            for j in range(i - 14, i):
                tr = max(
                    candles[j].high - candles[j].low,
                    abs(candles[j].high - candles[j-1].close) if j > 0 else 0,
                    abs(candles[j].low - candles[j-1].close) if j > 0 else 0
                )
                tr_sum += tr
            atr_values.append(tr_sum / 14)
        
        if not atr_values:
            return 50.0
            
        # Find percentile
        atr_values.sort()
        count = sum(1 for atr in atr_values if atr < current_atr)
        return (count / len(atr_values)) * 100
    
    def _classify_volatility_regime(self, realized_vol: float) -> str:
        """Classify volatility regime"""
        if realized_vol < 0.05:  # 5% annualized
            return "Ultra Low"
        elif realized_vol < 0.10:  # 10%
            return "Low"
        elif realized_vol < 0.20:  # 20%
            return "Normal"
        elif realized_vol < 0.30:  # 30%
            return "High"
        else:
            return "Extreme"
    
    def _assess_correlation_risk(self, symbol: str) -> str:
        """Assess correlation risk with other majors"""
        # Simplified correlation assessment
        if "USD" in symbol:
            return "High (USD exposure)"
        elif "EUR" in symbol or "GBP" in symbol:
            return "Medium (European exposure)"
        elif "JPY" in symbol or "CHF" in symbol:
            return "Medium (Safe haven)"
        else:
            return "Low"
    
    def _calculate_win_loss_stats(self, candles: List) -> tuple:
        """Calculate win rate and average win/loss"""
        if len(candles) < 50:
            return 0.5, 15.0, 10.0  # Defaults
            
        trades = []
        
        # Simulate simple momentum trades
        for i in range(20, len(candles) - 20):
            if candles[i].close > candles[i-1].close:  # Buy signal
                entry = candles[i].close
                exit = candles[i+10].close  # Exit after 10 bars
                pnl = (exit - entry) * 10000
                trades.append(pnl)
        
        if not trades:
            return 0.5, 15.0, 10.0
            
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0.5
        avg_win = sum(wins) / len(wins) if wins else 15.0
        avg_loss = sum(losses) / len(losses) if losses else 10.0
        
        return win_rate, avg_win, avg_loss
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0.02  # Default 2%
            
        # Kelly formula: f = p - q/b
        # p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 1
        
        kelly = p - (q / b)
        
        # Apply Kelly fraction (25% of full Kelly for safety)
        kelly = kelly * 0.25
        
        # Cap between 1% and 10%
        return max(0.01, min(0.10, kelly))
    
    def _find_technical_stop_levels(self, candles: List) -> List[float]:
        """Find natural technical stop levels"""
        if len(candles) < 20:
            return []
            
        levels = []
        
        # Recent swing lows for long stops
        for i in range(5, min(20, len(candles) - 5)):
            idx = len(candles) - i
            if idx >= 5:
                if candles[idx].low == min(c.low for c in candles[idx-5:idx+5]):
                    levels.append(candles[idx].low)
        
        # Recent swing highs for short stops  
        for i in range(5, min(20, len(candles) - 5)):
            idx = len(candles) - i
            if idx >= 5:
                if candles[idx].high == max(c.high for c in candles[idx-5:idx+5]):
                    levels.append(candles[idx].high)
        
        # Sort by distance from current price
        current = candles[-1].close
        return sorted(list(set(levels)), key=lambda x: abs(x - current))[:5]
    
    def _calculate_range(self, candles: List) -> float:
        """Calculate price range"""
        if not candles:
            return 0
        return max(c.high for c in candles) - min(c.low for c in candles)
    
    def _calculate_price_position(self, candles: List) -> float:
        """Calculate where price sits in recent range"""
        if len(candles) < 20:
            return 50.0
            
        recent = candles[-20:]
        high = max(c.high for c in recent)
        low = min(c.low for c in recent)
        current = candles[-1].close
        
        if high == low:
            return 50.0
            
        return ((current - low) / (high - low)) * 100
    
    def _is_position_trading_mode(self, market_data: Dict[str, MarketData]) -> bool:
        """Detect if using daily/weekly timeframes for position trading"""
        return 'd1' in market_data or 'w1' in market_data
    
    def _analyze_position_risk(self, market_data: Dict[str, MarketData],
                              news_context: Optional[List[str]],
                              ml_context: Optional[Dict[str, Any]]) -> AgentAnalysis:
        """Analyze risk for position trading with wider stops and longer timeframes"""
        
        d1_data = market_data.get('d1')
        w1_data = market_data.get('w1')
        h4_data = market_data.get('h4')  # For refined entry
        
        if not d1_data:
            raise ValueError("Daily data required for position risk analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            import numpy as np
            analyzer = ProfessionalMarketAnalyzer()
            
            # Need extensive history for position risk metrics
            d1_candles = d1_data.candles[-250:] if len(d1_data.candles) >= 250 else d1_data.candles
            w1_candles = w1_data.candles[-52:] if w1_data and len(w1_data.candles) >= 52 else []
            
            # Position trading volatility metrics
            realized_vol = analyzer.calculate_realized_volatility(d1_candles, 20)
            regime = analyzer.calculate_market_regime(d1_candles)
            
            use_enhanced = True
        except:
            use_enhanced = False
            realized_vol = 0.15  # Default 15% annual vol
            regime = {'regime': 'unknown'}
        
        current_price = d1_data.latest_candle.close
        daily_atr = d1_data.latest_candle.atr14
        spread = d1_data.latest_candle.spread or 0.0001
        
        # Position-specific risk calculations
        candles = d1_data.candles
        
        # Multi-day VaR and CVaR
        var_95_5d = self._calculate_multi_day_var(candles, 0.95, 5)
        cvar_95_5d = self._calculate_multi_day_cvar(candles, 0.95, 5)
        
        # Maximum drawdown analysis
        max_dd = self._calculate_max_drawdown(candles[-100:])
        
        # Position stop calculation (3-5x daily ATR)
        position_stop_distance = self._calculate_position_stop_loss(daily_atr, candles)
        
        # Correlation risk for multi-position exposure
        correlation_risk = self._assess_position_correlation_risk(d1_data.symbol)
        
        # Kelly for position sizing
        win_rate, avg_win, avg_loss = self._calculate_position_win_loss_stats(candles)
        kelly_fraction = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Weekly volatility regime
        weekly_vol_regime = self._classify_weekly_volatility(w1_data) if w1_data else "Unknown"
        
        # Gap risk assessment
        gap_risk = self._assess_gap_risk(candles)
        
        # Time decay risk (for time-based exits)
        time_decay_risk = self._assess_time_decay_risk(candles)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} risk for POSITION TRADING (multi-day/week holding).

POSITION VOLATILITY METRICS (250 daily bars):
- Daily ATR(14): {atr_pct:.2%} of price
- Realized Vol (annual): {realized_vol:.1%}
- Weekly Vol Regime: {weekly_vol_regime}
- Market Regime: {regime}

POSITION RISK METRICS:
- 5-day VaR (95%): {var_5d:.2%}
- 5-day CVaR (95%): {cvar_5d:.2%}
- Max Drawdown (100d): {max_dd:.2%}
- Gap Risk: {gap_risk}
- Spread Impact: {spread_impact:.3%}

POSITION SIZING:
- Account Risk Limit: ${max_risk:.2f} ({risk_percent}%)
- Kelly Fraction: {kelly:.1%}
- Win Rate (daily): {win_rate:.1%}
- Avg Win/Loss: {avg_win:.2%}/{avg_loss:.2%}

POSITION STOP ANALYSIS:
- Suggested Stop: {stop_distance:.1f} pips ({stop_pct:.2%})
- ATR Multiple: {atr_multiple:.1f}x
- Natural Support: {natural_support:.5f}
- Time Decay Risk: {time_decay}

PORTFOLIO CONSIDERATIONS:
- Correlation Risk: {corr_risk}
- Current Exposure: {current_exposure}
- Max Position Size: {max_position}

Position Risk Assessment Required:
1. Wide stop placement for multi-day volatility
2. Position size for extended holding period
3. Correlation with existing positions
4. Weekend/gap risk management
5. Time-based exit recommendations
6. Scaling in/out strategy

Provide:
POSITION_RISK: [Low/Medium/High/Extreme]
POSITION_SIZE: [0.01-0.5 lots for position trade]
STOP_DISTANCE: [Daily ATR multiple]
POSITION_TARGETS: [3 targets with risk multiples]
MAX_HOLDING: [Days before time stop]
SCALING_PLAN: [Entry and exit scaling]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
KEY_RISKS: [Main position trading risks]
GAP_PROTECTION: [Weekend/news gap strategy]

Format your response with exact labels above.
""",
            symbol=d1_data.symbol,
            atr_pct=daily_atr / current_price,
            realized_vol=realized_vol,
            weekly_vol_regime=weekly_vol_regime,
            regime=regime.get('regime', 'unknown'),
            var_5d=var_95_5d,
            cvar_5d=cvar_95_5d,
            max_dd=max_dd,
            gap_risk=gap_risk,
            spread_impact=(spread / current_price),
            max_risk=self.max_risk_per_trade,
            risk_percent=self.risk_per_trade * 100,
            kelly=kelly_fraction,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            stop_distance=position_stop_distance['distance'] * 10000,
            stop_pct=position_stop_distance['distance'] / current_price,
            atr_multiple=position_stop_distance['atr_multiple'],
            natural_support=position_stop_distance['support_level'],
            time_decay=time_decay_risk,
            corr_risk=correlation_risk,
            current_exposure="Check portfolio",  # Would need portfolio service
            max_position=f"{kelly_fraction * 100:.1f}% of capital"
        )
        
        # Risk Manager uses GPT-4 for position risk assessment
        from config.settings import get_settings
        settings = get_settings()
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.15,  # Even more conservative for position trades
            model_override=settings.gpt.risk_manager_model,
            agent_type=self.agent_type.value,
            symbol=d1_data.symbol
        )
        
        return self._parse_position_risk(response, current_price)
    
    def _calculate_multi_day_var(self, candles: List, confidence: float, days: int) -> float:
        """Calculate multi-day Value at Risk as percentage"""
        if len(candles) < days * 2:
            return 0.05 * days  # Default 5% per day
        
        # Calculate multi-day returns
        returns = []
        for i in range(days, len(candles)):
            multi_day_return = (candles[i].close - candles[i-days].close) / candles[i-days].close
            returns.append(multi_day_return)
        
        if not returns:
            return 0.05 * days
        
        # Sort and find VaR
        returns.sort()
        index = int((1 - confidence) * len(returns))
        return abs(returns[index]) if index < len(returns) else abs(returns[0])
    
    def _calculate_multi_day_cvar(self, candles: List, confidence: float, days: int) -> float:
        """Calculate multi-day Conditional VaR as percentage"""
        if len(candles) < days * 2:
            return 0.06 * days
        
        returns = []
        for i in range(days, len(candles)):
            multi_day_return = (candles[i].close - candles[i-days].close) / candles[i-days].close
            returns.append(multi_day_return)
        
        if not returns:
            return 0.06 * days
        
        returns.sort()
        index = int((1 - confidence) * len(returns))
        
        # Average of returns worse than VaR
        tail_returns = returns[:index] if index > 0 else returns[:1]
        return abs(sum(tail_returns) / len(tail_returns)) if tail_returns else 0.06 * days
    
    def _calculate_max_drawdown(self, candles: List) -> float:
        """Calculate maximum drawdown as percentage"""
        if len(candles) < 2:
            return 0.1  # Default 10%
        
        peak = candles[0].high
        max_dd = 0
        
        for candle in candles:
            # Update peak
            if candle.high > peak:
                peak = candle.high
            
            # Calculate drawdown from peak
            dd = (peak - candle.low) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_position_stop_loss(self, daily_atr: float, candles: List) -> Dict:
        """Calculate position stop loss (wider for multi-day holding)"""
        if len(candles) < 50:
            return {
                'distance': daily_atr * 4,
                'atr_multiple': 4,
                'support_level': candles[-1].close - daily_atr * 4
            }
        
        # Find major support level
        support_levels = []
        for i in range(20, len(candles) - 5):
            # Major swing low (20 bars each side for position trading)
            if candles[i].low == min(c.low for c in candles[i-20:i+5]):
                support_levels.append(candles[i].low)
        
        current_price = candles[-1].close
        
        # Filter support levels below current price
        valid_supports = [s for s in support_levels if s < current_price]
        
        if valid_supports:
            # Use nearest major support
            nearest_support = max(valid_supports)
            support_distance = current_price - nearest_support
            
            # Ensure minimum 3x ATR
            if support_distance < daily_atr * 3:
                distance = daily_atr * 3.5
                support = current_price - distance
            else:
                distance = support_distance * 1.1  # Add 10% buffer
                support = nearest_support * 0.99  # Just below support
        else:
            # Default to 4x ATR for position trades
            distance = daily_atr * 4
            support = current_price - distance
        
        return {
            'distance': distance,
            'atr_multiple': distance / daily_atr,
            'support_level': support
        }
    
    def _assess_position_correlation_risk(self, symbol: str) -> str:
        """Assess correlation risk for position trading portfolio"""
        # Enhanced correlation assessment for position trades
        base_currency = symbol[:3]
        quote_currency = symbol[3:6] if len(symbol) >= 6 else "XXX"
        
        risk_factors = []
        
        if "USD" in symbol:
            risk_factors.append("USD exposure")
        if "EUR" in symbol or "GBP" in symbol:
            risk_factors.append("European exposure")
        if "JPY" in symbol or "CHF" in symbol:
            risk_factors.append("Safe haven exposure")
        if "AUD" in symbol or "NZD" in symbol or "CAD" in symbol:
            risk_factors.append("Commodity currency")
        
        if len(risk_factors) >= 2:
            return f"High ({', '.join(risk_factors)})"
        elif risk_factors:
            return f"Medium ({risk_factors[0]})"
        else:
            return "Low (exotic pair)"
    
    def _calculate_position_win_loss_stats(self, candles: List) -> tuple:
        """Calculate win/loss stats for position trading (multi-day holds)"""
        if len(candles) < 100:
            return 0.45, 0.03, 0.02  # Conservative defaults
        
        trades = []
        
        # Simulate position trades (10-day holds)
        for i in range(50, len(candles) - 10):
            # Entry on breakout
            if candles[i].close > max(c.high for c in candles[i-20:i]):
                entry = candles[i].close
                exit = candles[i+10].close  # Exit after 10 days
                pnl = (exit - entry) / entry
                trades.append(pnl)
        
        if not trades:
            return 0.45, 0.03, 0.02
        
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0.45
        avg_win = sum(wins) / len(wins) if wins else 0.03
        avg_loss = sum(losses) / len(losses) if losses else 0.02
        
        return win_rate, avg_win, avg_loss
    
    def _classify_weekly_volatility(self, weekly_data: Optional[MarketData]) -> str:
        """Classify volatility on weekly timeframe"""
        if not weekly_data or len(weekly_data.candles) < 20:
            return "Unknown"
        
        # Calculate weekly ATR as percentage
        candles = weekly_data.candles
        weekly_ranges = []
        
        for i in range(1, min(20, len(candles))):
            weekly_range = (candles[i].high - candles[i].low) / candles[i].close
            weekly_ranges.append(weekly_range)
        
        avg_weekly_range = sum(weekly_ranges) / len(weekly_ranges) if weekly_ranges else 0.02
        
        if avg_weekly_range < 0.01:  # Less than 1% weekly
            return "Ultra Low"
        elif avg_weekly_range < 0.02:  # Less than 2%
            return "Low"
        elif avg_weekly_range < 0.04:  # Less than 4%
            return "Normal"
        elif avg_weekly_range < 0.06:  # Less than 6%
            return "High"
        else:
            return "Extreme"
    
    def _assess_gap_risk(self, candles: List) -> str:
        """Assess weekend and overnight gap risk"""
        if len(candles) < 50:
            return "Unknown"
        
        gaps = []
        for i in range(1, len(candles)):
            gap = abs(candles[i].open - candles[i-1].close) / candles[i-1].close
            if gap > 0.002:  # More than 0.2%
                gaps.append(gap)
        
        if not gaps:
            return "Low - No significant gaps"
        
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        
        if max_gap > 0.02:  # 2% gap
            return f"High - Max gap {max_gap:.1%}"
        elif avg_gap > 0.005:  # 0.5% average
            return f"Medium - Avg gap {avg_gap:.1%}"
        else:
            return "Low - Minor gaps only"
    
    def _assess_time_decay_risk(self, candles: List) -> str:
        """Assess risk of position decay over time"""
        if len(candles) < 100:
            return "Unknown"
        
        # Check how often positions reverse after X days
        reversals_5d = 0
        reversals_10d = 0
        reversals_20d = 0
        
        for i in range(20, len(candles) - 20):
            # Check if trend reverses
            initial_trend = 1 if candles[i].close > candles[i-5].close else -1
            
            if len(candles) > i + 5:
                trend_5d = 1 if candles[i+5].close > candles[i].close else -1
                if trend_5d != initial_trend:
                    reversals_5d += 1
            
            if len(candles) > i + 10:
                trend_10d = 1 if candles[i+10].close > candles[i].close else -1
                if trend_10d != initial_trend:
                    reversals_10d += 1
            
            if len(candles) > i + 20:
                trend_20d = 1 if candles[i+20].close > candles[i].close else -1
                if trend_20d != initial_trend:
                    reversals_20d += 1
        
        total_samples = len(candles) - 40
        if total_samples <= 0:
            return "Unknown"
        
        reversal_rate_10d = reversals_10d / total_samples
        
        if reversal_rate_10d > 0.6:
            return "High - Trends rarely last >10 days"
        elif reversal_rate_10d > 0.4:
            return "Medium - Mixed persistence"
        else:
            return "Low - Trends tend to persist"
    
    def _parse_position_risk(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse position risk analysis response"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            raw_parsed = parsed_data['metadata']['raw_parsed']
            
            # Position risk is more conservative
            position_risk = raw_parsed.get('POSITION_RISK', 'High').upper()
            
            if 'HIGH' in position_risk or 'EXTREME' in position_risk:
                recommendation = SignalType.WAIT
                confidence = 95.0  # Very high confidence in waiting
            else:
                rec_text = raw_parsed.get('RECOMMENDATION', 'WAIT').upper()
                if 'BUY' in rec_text and 'LOW' in position_risk:
                    recommendation = SignalType.BUY
                elif 'SELL' in rec_text and 'LOW' in position_risk:
                    recommendation = SignalType.SELL
                else:
                    recommendation = SignalType.WAIT
                
                confidence = parsed_data['confidence']
            
            # Extract position-specific risk metrics
            try:
                position_size = float(raw_parsed.get('POSITION_SIZE', '0.1').replace('lots', '').strip())
            except:
                position_size = 0.1
            
            try:
                stop_distance = float(raw_parsed.get('STOP_DISTANCE', '4').replace('x', '').strip())
            except:
                stop_distance = 4.0
            
            # Parse position targets
            targets_str = raw_parsed.get('POSITION_TARGETS', '')
            targets = []
            if targets_str:
                import re
                # Match patterns like "1.2345 (2R)"
                matches = re.findall(r'([\d.]+)\s*\(([^)]+)\)', targets_str)
                for price, risk_multiple in matches:
                    targets.append({'price': float(price), 'risk_multiple': risk_multiple})
            
            # Calculate position stop loss
            if recommendation == SignalType.BUY:
                stop_loss = current_price * (1 - stop_distance * 0.01)
                take_profit = targets[0]['price'] if targets else current_price * 1.03
            elif recommendation == SignalType.SELL:
                stop_loss = current_price * (1 + stop_distance * 0.01)
                take_profit = targets[0]['price'] if targets else current_price * 0.97
            else:
                stop_loss = 0
                take_profit = 0
            
            # Build reasoning
            reasoning = [
                f"Position risk: {position_risk}",
                f"Position size: {position_size} lots",
                f"Stop: {stop_distance}x daily ATR"
            ]
            
            concerns = []
            if raw_parsed.get('KEY_RISKS'):
                concerns.append(raw_parsed['KEY_RISKS'])
            
            # Update metadata
            parsed_data['metadata'].update({
                'position_risk': position_risk,
                'position_size': position_size,
                'stop_atr_multiple': stop_distance,
                'max_holding': raw_parsed.get('MAX_HOLDING', '20 days'),
                'scaling_plan': raw_parsed.get('SCALING_PLAN', 'Single entry'),
                'gap_protection': raw_parsed.get('GAP_PROTECTION', 'Reduce before weekend'),
                'position_targets': targets,
                'position_mode': True
            })
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                concerns=concerns,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self._calculate_risk_reward(
                    current_price,
                    stop_loss,
                    take_profit
                ) if stop_loss and take_profit else None,
                metadata=parsed_data['metadata']
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            # Position risk defaults to maximum safety
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=98.0,  # Almost certain to wait on error
                reasoning=["Position risk analysis failed - maximum safety protocol"],
                concerns=["Unable to assess position risk - trade rejected"],
                entry_price=current_price,
                metadata={
                    'position_risk': 'Extreme',
                    'error': str(e),
                    'position_mode': True
                }
            )
            self.analysis_history.append(default_analysis)
            return default_analysis