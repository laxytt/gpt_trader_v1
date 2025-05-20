import json

from core.paths import COMPLETED_TRADES_FILE

def evaluate_log(filepath=COMPLETED_TRADES_FILE):
    total, wins, losses = 0, 0, 0
    rr_buckets = {"A": [], "B": [], "C": []}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
            except Exception as e:
                print(f"⚠️ Error parsing trade: {e}")
                continue

            total += 1
            result = trade.get("result", "")
            if result == "TP_hit":
                wins += 1
            elif result == "SL_hit":
                losses += 1

            tier = trade.get("risk_class", "B")
            rr_buckets.setdefault(tier, []).append(result)

    if total == 0:
        print("No trades found.")
        return

    print(f"📊 Evaluated {total} trades.")
    print(f"✅ Wins: {wins}, ❌ Losses: {losses}, ⚖️ Win rate: {wins / total * 100:.1f}%")
    for tier, outcomes in rr_buckets.items():
        if not outcomes:
            continue
        wr = outcomes.count("TP_hit") / len(outcomes) * 100
        print(f"  • Risk {tier}: {len(outcomes)} trades, Win rate = {wr:.1f}%")

if __name__ == "__main__":
    evaluate_log()
