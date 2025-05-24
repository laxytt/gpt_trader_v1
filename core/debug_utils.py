import json

def print_section(title):
    print("\n" + "="*60)
    print(f"=== {title}")
    print("="*60)

def pretty_print_json_box(obj, title="GPT Output"):
    print(f"\n----- {title} -----")
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    print("-"*60)

def print_symbol_header(symbol, state):
    ticket = state.get('ticket', '?')
    status = state.get('status', '?')
    lots = state.get('lots', '?')
    side = state.get('side', '?')
    entry = state.get('entry', '?')
    print(f"\n----- {symbol} | State: {status} | Ticket: {ticket} | Side: {side} | Lots: {lots} | Entry: {entry} -----")
