# Symbol Resolution Guide

## Overview

The trading system now includes automatic symbol resolution to handle different broker naming conventions. This ensures compatibility across different MT5 brokers that may use different names for the same instruments (e.g., GOLD vs XAUUSD, US30 vs DJ30).

## Components

### 1. Symbol Resolver (`core/utils/symbol_resolver.py`)

The main component that:
- Maintains a cache of available symbols from MT5
- Maps common symbol names to broker-specific names
- Automatically enables symbols in Market Watch when needed
- Provides partial matching for unknown symbols

### 2. Symbol Mappings

Common mappings handled automatically:

**Metals:**
- GOLD → XAUUSD, Gold, XAU/USD, XAUUSDm, etc.
- SILVER → XAGUSD, Silver, XAG/USD, XAGUSDm, etc.

**Indices:**
- US30 → DJ30, US30Cash, USA30, Wall Street 30, etc.
- US500 → SP500, US500Cash, S&P500, SPX500, etc.
- NASDAQ → USTEC, NAS100, US100, TECH100, etc.
- DAX → GER30, GER40, DAX30, Germany30, etc.

**Energy:**
- OIL → XTIUSD, USOil, WTI, CrudeOil, etc.
- BRENT → XBRUSD, UKOil, BrentOil, etc.

**Crypto:**
- BITCOIN → BTCUSD, Bitcoin, BTC/USD, etc.
- ETHEREUM → ETHUSD, Ethereum, ETH/USD, etc.

## Integration Points

### MT5 Data Provider
Automatically resolves symbols when fetching market data:
```python
# Before
request = DataRequest(symbol=symbol, ...)

# After
resolved_symbol = self.symbol_resolver.resolve_and_enable(symbol)
request = DataRequest(symbol=resolved_symbol, ...)
```

### MT5 Order Manager
Resolves symbols when:
- Calculating lot sizes
- Getting current prices
- Placing orders

## Usage

### In Trading Loop

The system automatically handles symbol resolution. Just use common names in your configuration:

```python
# config/symbols.py
TRADING_SYMBOLS = {
    'conservative': ['EURUSD', 'GBPUSD', 'USDJPY'],
    'moderate': ['AUDUSD', 'NZDUSD', 'GOLD'],  # GOLD will be resolved to broker's name
    'aggressive': ['US30', 'OIL']  # These will be resolved automatically
}
```

### Manual Symbol Resolution

For testing or manual operations:

```python
from core.utils.symbol_resolver import get_symbol_resolver

resolver = get_symbol_resolver()

# Resolve a symbol
actual_symbol = resolver.resolve_symbol('GOLD')  # Returns 'XAUUSD' on most brokers

# Resolve and enable in Market Watch
enabled_symbol = resolver.resolve_and_enable('US30')  # Returns actual name and enables it

# Get all available symbols
all_symbols = resolver.get_available_symbols()
forex_symbols = resolver.get_available_symbols('Forex')
```

### Interactive Symbol Finder

Use the provided tools for finding and enabling symbols:

```bash
# Interactive mode
python test_symbol_finder.py

# Quick enable common symbols
python test_symbol_finder.py --quick

# Test MT5 connection with symbol resolution
python test_mt5_simple.py
```

## Troubleshooting

### Symbol Not Found

If a symbol is not found:
1. Use the interactive finder to search for variations
2. Check if the symbol is available on your broker
3. Add new mappings to `SYMBOL_MAPPINGS` in `symbol_resolver.py`

### Symbol Not Visible

If a symbol exists but isn't visible:
- The resolver automatically enables symbols in Market Watch
- Manual enable: `mt5.symbol_select(symbol_name, True)`

### Multiple Matches

When multiple symbols match:
- The resolver returns the shortest match (usually most common)
- Use the interactive finder to see all matches

## Best Practices

1. **Use Common Names**: Always use common symbol names (GOLD, US30) in configuration
2. **Let the System Resolve**: Don't hardcode broker-specific names
3. **Test New Symbols**: Use the interactive finder when adding new symbols
4. **Cache Efficiency**: The resolver caches symbols on initialization for performance

## Adding New Mappings

To add support for new symbol variations:

1. Edit `core/utils/symbol_resolver.py`
2. Add to `SYMBOL_MAPPINGS` dictionary:
```python
'YOUR_SYMBOL': ['Variant1', 'Variant2', 'Variant3'],
```

3. Test with the interactive finder:
```bash
python test_symbol_finder.py
```

## Performance Considerations

- Symbol resolution is cached for performance
- Resolution happens once per symbol per session
- Enabling symbols in Market Watch is done only when needed
- No performance impact on trading operations

## Error Handling

The system gracefully handles resolution failures:
- Falls back to original symbol name if resolution fails
- Logs warnings for failed resolutions
- Trading continues with original symbol name
- No interruption to trading operations