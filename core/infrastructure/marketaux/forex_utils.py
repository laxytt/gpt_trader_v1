"""Utilities for handling forex symbols with MarketAux API"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class ForexSymbolConverter:
    """Convert forex pair symbols to individual currencies for MarketAux"""
    
    # Standard forex pairs and their component currencies
    FOREX_PAIRS = {
        'EURUSD': ('EUR', 'USD'),
        'GBPUSD': ('GBP', 'USD'),
        'USDJPY': ('USD', 'JPY'),
        'USDCAD': ('USD', 'CAD'),
        'AUDUSD': ('AUD', 'USD'),
        'NZDUSD': ('NZD', 'USD'),
        'USDCHF': ('USD', 'CHF'),
        'EURJPY': ('EUR', 'JPY'),
        'EURGBP': ('EUR', 'GBP'),
        'GBPJPY': ('GBP', 'JPY'),
        'XAUUSD': ('XAU', 'USD'),  # Gold
        'XAGUSD': ('XAG', 'USD'),  # Silver
    }
    
    # Currency to country mapping for MarketAux countries parameter
    # Using lowercase country codes as per MarketAux documentation
    CURRENCY_COUNTRIES = {
        'EUR': ['de', 'fr', 'it', 'es', 'nl', 'be', 'at'],  # Major Eurozone countries
        'USD': ['us'],
        'GBP': ['gb'],  # Great Britain (MarketAux uses 'gb' not 'uk')
        'JPY': ['jp'],
        'CAD': ['ca'],
        'AUD': ['au'],
        'NZD': ['nz'],
        'CHF': ['ch'],
        'CNY': ['cn'],
        'HKD': ['hk'],
        'MXN': ['mx'],
        'ZAR': ['za'],
        'TRY': ['tr'],
        'RUB': ['ru'],
        'INR': ['in'],
        'BRL': ['br'],
        'KRW': ['kr'],
        'PLN': ['pl'],
        'XAU': [],  # Gold - no specific country
        'XAG': [],  # Silver - no specific country
    }
    
    @staticmethod
    def split_forex_pair(symbol: str) -> Optional[Tuple[str, str]]:
        """
        Split a forex pair symbol into its component currencies
        
        Args:
            symbol: Forex pair symbol (e.g., "EURUSD", "EUR/USD", "EUR-USD")
            
        Returns:
            Tuple of (base_currency, quote_currency) or None if not recognized
        """
        # Clean the symbol
        clean_symbol = symbol.upper().replace('/', '').replace('-', '').replace(' ', '')
        
        # Check if it's a known pair
        if clean_symbol in ForexSymbolConverter.FOREX_PAIRS:
            return ForexSymbolConverter.FOREX_PAIRS[clean_symbol]
        
        # Try to parse 6-character symbols (e.g., "EURUSD")
        if len(clean_symbol) == 6:
            base = clean_symbol[:3]
            quote = clean_symbol[3:]
            if base in ForexSymbolConverter.CURRENCY_COUNTRIES and quote in ForexSymbolConverter.CURRENCY_COUNTRIES:
                return (base, quote)
        
        # Check if it's already a single currency
        if clean_symbol in ForexSymbolConverter.CURRENCY_COUNTRIES:
            return (clean_symbol,)
        
        logger.warning(f"Unable to parse forex symbol: {symbol}")
        return None
    
    @staticmethod
    def get_currencies_for_symbol(symbol: str) -> List[str]:
        """
        Get list of currencies for a trading symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of currency codes
        """
        currencies = ForexSymbolConverter.split_forex_pair(symbol)
        if currencies:
            return list(currencies)
        return []
    
    @staticmethod
    def get_countries_for_symbol(symbol: str) -> List[str]:
        """
        Get list of countries relevant to a forex symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of country codes for MarketAux API
        """
        currencies = ForexSymbolConverter.split_forex_pair(symbol)
        if not currencies:
            return []
        
        countries = []
        for currency in currencies:
            country_list = ForexSymbolConverter.CURRENCY_COUNTRIES.get(currency, [])
            countries.extend(country_list)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_countries = []
        for country in countries:
            if country not in seen:
                seen.add(country)
                unique_countries.append(country)
        
        return unique_countries
    
    @staticmethod
    def get_marketaux_params_for_symbols(symbols: List[str], free_plan: bool = True) -> dict:
        """
        Convert forex symbols to MarketAux API parameters
        
        Args:
            symbols: List of trading symbols (e.g., ["EURUSD", "GBPUSD"])
            free_plan: Whether using free plan (affects strategy)
            
        Returns:
            Dict with 'symbols', 'countries', and 'search' parameters for MarketAux
        """
        all_currencies = []
        all_countries = []
        forex_pairs = []
        
        for symbol in symbols:
            # Get currencies for this symbol
            currencies = ForexSymbolConverter.get_currencies_for_symbol(symbol)
            all_currencies.extend(currencies)
            
            # Get countries for this symbol
            countries = ForexSymbolConverter.get_countries_for_symbol(symbol)
            all_countries.extend(countries)
            
            # Keep original forex pairs
            if len(currencies) == 2:
                forex_pairs.append(symbol)
        
        # Remove duplicates
        unique_currencies = list(dict.fromkeys(all_currencies))
        unique_countries = list(dict.fromkeys(all_countries))
        unique_forex_pairs = list(dict.fromkeys(forex_pairs))
        
        params = {}
        
        if free_plan:
            # Free plan: Focus on highly targeted forex search queries
            # Don't use symbols parameter as it returns 0 results
            
            # Use major forex market countries only
            forex_countries = []
            for currency in unique_currencies:
                if currency == 'USD':
                    forex_countries.append('us')
                elif currency == 'EUR':
                    forex_countries.extend(['de', 'fr'])  # Major EU economies
                elif currency == 'GBP':
                    forex_countries.append('gb')
                elif currency == 'JPY':
                    forex_countries.append('jp')
            
            if forex_countries:
                params['countries'] = list(dict.fromkeys(forex_countries))[:3]  # Top 3 unique countries
            
            # Build highly targeted forex search query
            search_terms = []
            
            # Primary forex-specific terms (most important)
            # Use more targeted terms that are likely to appear in forex news
            search_terms.extend([
                '"forex"',
                '"fx market"',
                '"currency"',
                '"exchange rate"'
            ])
            
            # Add specific pair mentions with slash notation
            for pair in unique_forex_pairs[:2]:  # Top 2 pairs
                if len(pair) == 6:
                    formatted_pair = f"{pair[:3]}/{pair[3:]}"
                    search_terms.append(f'"{formatted_pair}"')
            
            # Currency-specific search terms
            currency_search_terms = {
                'USD': '"US dollar" | "dollar index" | DXY',
                'EUR': '"euro currency" | "eurozone" | "EUR USD"',
                'GBP': '"pound sterling" | "cable" | "GBP USD"',
                'JPY': '"japanese yen" | "USD JPY" | "yen crosses"',
                'AUD': '"aussie dollar" | "AUD USD"',
                'CAD': '"canadian dollar" | "USD CAD"',
                'CHF': '"swiss franc" | "safe haven"',
                'NZD': '"kiwi dollar" | "NZD USD"'
            }
            
            # Add currency-specific terms for top 2 currencies
            for currency in unique_currencies[:2]:
                if currency in currency_search_terms:
                    search_terms.append(f'({currency_search_terms[currency]})')
            
            # Central bank and economic data terms
            central_bank_terms = []
            if 'USD' in unique_currencies:
                central_bank_terms.extend(['fed', 'fomc', '"federal reserve"'])
            if 'EUR' in unique_currencies:
                central_bank_terms.extend(['ecb', '"european central bank"'])
            if 'GBP' in unique_currencies:
                central_bank_terms.extend(['boe', '"bank of england"'])
            if 'JPY' in unique_currencies:
                central_bank_terms.extend(['boj', '"bank of japan"'])
            
            if central_bank_terms:
                search_terms.append(f'({" | ".join(central_bank_terms[:3])})')
            
            # Economic indicators
            search_terms.append('("interest rate" | inflation | nfp | "non-farm payroll" | gdp | cpi)')
            
            # Join with OR operator, prioritizing most specific terms
            params['search'] = ' | '.join(search_terms)
            
        else:
            # Paid plan: Can use symbols and entities
            if unique_currencies:
                params['symbols'] = unique_currencies
                
            if unique_countries:
                params['countries'] = unique_countries
            
            # Still use search for better results
            search_terms = ['forex', 'currency', '"exchange rate"', '"central bank"']
            
            # Add currency codes
            for currency in unique_currencies[:4]:
                search_terms.append(currency)
                
            # Add forex pairs
            for pair in unique_forex_pairs[:3]:
                search_terms.append(f'"{pair}"')
            
            params['search'] = ' | '.join(search_terms)
        
        return params
    
    @staticmethod
    def is_forex_symbol(symbol: str) -> bool:
        """Check if a symbol is a forex pair"""
        clean_symbol = symbol.upper().replace('/', '').replace('-', '').replace(' ', '')
        return clean_symbol in ForexSymbolConverter.FOREX_PAIRS or len(clean_symbol) == 6