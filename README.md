📈 GPT VSA Trading System — Complete Project Workflow

Step-by-Step Process (with RAG, Reflection, Memory)
1. Loop Start & Data Preparation
Start main loop: Wait for next M5 candle close.

Data pre-checks: News filter, calendar/data freshness, screenshot freshness.

Sync state: Load trade status, check MT5 for live trades. Heal if out-of-sync/corrupted.

2. If Trade is Open: Management Branch
Fetch M5 & H1 data, context, news.

Run GPT management:

Inputs: open trade info, M5/H1 context, news, performance streak, feedback.

Actions:

HOLD: Keep position, maybe move SL to breakeven (auto if conditions met).

MOVE_SL: Adjust stop-loss per model logic (e.g., ATR or BE).

CLOSE_NOW: Immediate close if risk/trend/news/timeout triggers.

If trade is closed:

Log trade outcome.

Add case to RAG memory (rag_memory.py) for future prompt retrieval.

Run GPT Reflection:

Inputs: trade summary, both M5/H1 context at trade time, outcome, context.

GPT provides a human-style review (was the signal valid? How to improve?).

Append reflection to trade record.

Return state to idle.

3. If No Trade is Open: Signal Generation Branch
Fetch M5 & H1 data, screenshots, news, context, and win/loss streak.

Build prompt for GPT, including:

M5 and H1 candles (with indicators)

Macroeconomic events

RAG: Up to 3 most similar historical trade cases retrieved from memory

Session/volatility, performance context

Run GPT signal:

WAIT: No action; log, loop resumes.

BUY/SELL:

Attempt to open trade in MT5.

On success:

Update state to open.

Trade is managed in next loop.

On failure:

Log error, skip.

4. RAG Memory and Case Logging
After every closed trade:

Format a trade case (indicators, context, signal, RR, reason, result, reflection).

Add to RAG vector store:

Used for semantic retrieval in future prompts to help GPT “learn from experience.”

RAG used in every signal cycle:

Query most similar past trade situations for context/risk calibration.

Passed into prompt to reinforce rule-based, data-driven learning.

5. Reflection Process
After every closed trade:

GPT reviews trade outcome in context of strict VSA rules and multi-timeframe logic.

Generates a brief reflection paragraph:

Was signal valid?

Did trade follow all rules (background, confirmation, risk, news filter)?

What could be improved?

Reflection is logged to file/memory for future analysis.

6. Logging & Notifications
All trades, outcomes, management actions, errors are logged.

Optionally, key events and errors are sent to Telegram for real-time monitoring.

All Scenarios
Trade Open:

Managed every cycle by GPT, using current M5/H1 context, news, and RAG.

Possible actions: HOLD, MOVE_SL, CLOSE_NOW.

Any closure triggers logging, RAG memory update, and reflection.

No Trade Open:

GPT analyzes context, retrieves past similar cases, and signals WAIT, BUY, or SELL.

WAIT → skip.

BUY/SELL → order sent; on success, tracked as open position.

News or data not fresh:

Skips trading until safe.

Corrupted/missing files:

Attempts auto-heal/reset, always logs.

Errors/failures:

Always logs and skips cycle (never crashes loop).

[Main Loop]
   │
   ├─► [Pre-checks: News, Calendar, Data Freshness]
   │
   ├─► [State Management (MT5 sync)]
   │
   ├─► Trade Open?
   │       │
   │    Yes│No
   │       ▼
   │ [Trade Management (GPT)]
   │   │      │
   │ HOLD  MOVE_SL  CLOSE_NOW
   │   │      │         │
   │   └──────┴─────────┘
   │          │
   │   [If closed:]
   │      │
   │   [Log, Add to RAG, Reflect]
   │      │
   │   [State Idle]
   │
   └─► [Signal Generation (GPT)]
           │
     ┌─────┼─────┐
   WAIT  BUY/SELL
     │      │
   Loop   [Open trade in MT5]
            │
         [Success?]
         │       │
       Yes      No
       │         │
   [State Open]  [Log, skip]
