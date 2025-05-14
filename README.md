# GPT Trader v1

System do automatycznej analizy świec i generowania sygnałów tradingowych z użyciem GPT-4o/4.1 oraz integracją z MT5.

## Zawartość:
- `main_cycle.py` – główna pętla decyzyjna
- `gpt_interface.py` – komunikacja z ChatGPT
- `mt5_utils.py` – pomocnicze funkcje MT5
- `trade_status.json` – zapis stanu pozycji
- `gpt_signals_log.jsonl` – historia sygnałów GPT
- `closed_trades.jsonl` – historia zamkniętych transakcji

## Wymagania:
- MetaTrader5 (zainstalowany)
- Python 3.9+
- Biblioteki: openai, MetaTrader5, pandas, tiktoken

## Uruchomienie:
```
python main_cycle.py
```
