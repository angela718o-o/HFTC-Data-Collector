import shift
import pandas as pd
import time
from datetime import datetime
import os

TICKER = "AAPL"  # symbol to collect
CFG_FILE = "./initiator.cfg"  # FIX config file (must exist)
USERNAME = "four-sigma"
PASSWORD = "sk8Pf6nJ"
INTERVAL = 0.5  # seconds between snapshots
DEPTH = 5       # order book depth levels to store
OUTPUT_FILE = "order_book.csv"

if not os.path.exists(CFG_FILE):
    raise FileNotFoundError(f"Config file not found: {CFG_FILE}")

trader = shift.Trader(username=USERNAME)
connected = trader.connect(cfg_file=CFG_FILE, password=PASSWORD)
if not connected:
    raise Exception("Could not connect to SHIFT.")
print("Connected to SHIFT")

success = trader.sub_order_book(TICKER)
if not success:
    raise Exception(f"Could not subscribe to order book for {TICKER}")
print(f"Subscribed to {TICKER} order book")


data = []

def extract_features(book):
    row = {}
    best_bid = book['bids'][0].price if book['bids'] else 0
    best_ask = book['asks'][0].price if book['asks'] else 0

    row["bid"] = best_bid
    row["ask"] = best_ask
    row["mid"] = (best_bid + best_ask)/2 if (best_bid + best_ask) > 0 else 0
    row["spread"] = best_ask - best_bid if best_bid and best_ask else 0

    # Top-DEPTH bids
    for i in range(min(DEPTH, len(book['bids']))):
        row[f"bid_{i}_price"] = book['bids'][i].price
        row[f"bid_{i}_size"] = book['bids'][i].size

    # Top-DEPTH asks
    for i in range(min(DEPTH, len(book['asks']))):
        row[f"ask_{i}_price"] = book['asks'][i].price
        row[f"ask_{i}_size"] = book['asks'][i].size

    # Order book imbalance
    bid_vol = sum([b.size for b in book['bids'][:DEPTH]])
    ask_vol = sum([a.size for a in book['asks'][:DEPTH]])
    row["imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

    return row

try:
    print("Collecting data... Press Ctrl+C to stop.")
    while True:
        book_obj_bid = trader.get_order_book(TICKER,shift.OrderBookType.GLOBAL_BID, max_level=DEPTH)
        book_obj_asks=trader.get_order_book(TICKER,shift.OrderBookType.GLOBAL_ASK, max_level=DEPTH)
        book = {"bids": book_obj_bid, "asks": book_obj_asks}

        row = extract_features(book)
        row["timestamp"] = datetime.now()

        data.append(row)
        print(f"{row['timestamp']} | mid={row['mid']:.2f} | spread={row['spread']:.4f}")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n⏹ Stopping collector...")

finally:
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(data)} rows to {OUTPUT_FILE}")

    trader.disconnect()
    print("Disconnected from SHIFT")