
from core.database import memory

def get_win_loss_streak(symbol="EURUSD", sample_size=10):
    try:
        cases = memory.query_cases(symbol=symbol, limit=sample_size)
        results = [case.get("result", "").lower() for case in cases]

        win_count = sum("win" in res for res in results)
        win_rate = win_count / len(results) if results else 0.0

        streak_type = "N/A"
        streak_length = 0
        if results:
            last_type = "win" if "win" in results[-1] else "loss"
            streak_type = last_type
            streak_length = 1
            for res in reversed(results[:-1]):
                this_type = "win" if "win" in res else "loss"
                if this_type == last_type:
                    streak_length += 1
                else:
                    break

        return {
            "streak_type": streak_type,
            "streak_length": streak_length,
            "win_rate": win_rate,
            "sample_size": sample_size
        }

    except Exception as e:
        return {
            "streak_type": "N/A",
            "streak_length": 0,
            "win_rate": 0.0,
            "sample_size": sample_size
        }
