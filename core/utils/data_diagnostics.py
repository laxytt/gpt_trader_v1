# core/utils/data_diagnostics.py
"""Diagnostic utilities for data availability and quality"""

from asyncio.log import logger
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from core.domain.enums.mt5_enums import TimeFrame
from core.domain.models import MarketData
from core.infrastructure.data.unified_data_provider import DataRequest, UnifiedDataProvider


class DataDiagnostics:
    """Provides comprehensive data diagnostics and reporting"""
    
    def __init__(self, unified_provider: UnifiedDataProvider):
        self.provider = unified_provider
    
    async def generate_availability_report(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive data availability report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbols': {}
        }
        
        for symbol in symbols:
            try:
                start_date, end_date = self.provider.get_available_date_range(symbol)
                
                report['symbols'][symbol] = {
                    'available': True,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'total_days': (end_date - start_date).days,
                    'timeframes': {}
                }
                
                # Check each timeframe
                for tf in [TimeFrame.M1, TimeFrame.H1, TimeFrame.D1]:
                    try:
                        test_request = DataRequest(
                            symbol=symbol,
                            timeframe=tf,
                            num_bars=100
                        )
                        data = await self.provider.get_data(test_request)
                        
                        report['symbols'][symbol]['timeframes'][tf.value] = {
                            'available': True,
                            'bars_available': len(data.candles),
                            'has_indicators': any(c.ema50 is not None for c in data.candles)
                        }
                    except Exception as e:
                        report['symbols'][symbol]['timeframes'][tf.value] = {
                            'available': False,
                            'error': str(e)
                        }
                        
            except Exception as e:
                report['symbols'][symbol] = {
                    'available': False,
                    'error': str(e)
                }
        
        return report
    
    def log_data_request_summary(
        self,
        request: DataRequest,
        result: Optional[MarketData],
        error: Optional[Exception] = None
    ):
        """Log comprehensive summary of data request"""
        if result:
            logger.info(
                f"Data request successful - Symbol: {request.symbol}, "
                f"Timeframe: {request.timeframe.value}, "
                f"Bars returned: {len(result.candles)}, "
                f"Date range: {result.candles[0].timestamp} to {result.candles[-1].timestamp}"
            )
        else:
            logger.error(
                f"Data request failed - Symbol: {request.symbol}, "
                f"Timeframe: {request.timeframe.value}, "
                f"Requested: {request.start_date} to {request.end_date}, "
                f"Error: {error}"
            )