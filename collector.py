import shift
import pandas as pd
import time
from datetime import datetime
import os

TICKER = "CS1"  # symbol to collect
CFG_FILE = "./initiator.cfg"  
USERNAME = "four-sigma"
PASSWORD = "sk8Pf6nJ"
INTERVAL = 0.5  
DEPTH = 5       
OUTPUT_FILE = "order_book.csv"

if not os.path.exists(CFG_FILE):
    raise FileNotFoundError(f"Config file not found: {CFG_FILE}")

trader = shift.Trader(username=USERNAME)
connected = trader.connect(cfg_file=CFG_FILE, password=PASSWORD)
if not connected:
    raise Exception("Could not connect to SHIFT.")
print("Connected to SHIFT")

# 修正点 1: 建议订阅所有股票，因为 local 模式下流动性非常分散
trader.sub_all_order_book()
print(f"Subscribed to Local Order Books")

data = []

def extract_features(book):
    row = {}
    best_bid = book['bids'][0].price if book['bids'] else 0
    best_ask = book['asks'][0].price if book['asks'] else 0

    row["bid"] = best_bid
    row["ask"] = best_ask
    row["mid"] = (best_bid + best_ask)/2 if (best_bid + best_ask) > 0 else 0
    row["spread"] = best_ask - best_bid if best_bid and best_ask else 0

    # 提取买单深度
    for i in range(DEPTH):
        if i < len(book['bids']):
            row[f"bid_{i}_price"] = book['bids'][i].price
            row[f"bid_{i}_size"] = book['bids'][i].size
        else:
            row[f"bid_{i}_price"], row[f"bid_{i}_size"] = 0, 0

    # 提取卖单深度
    for i in range(DEPTH):
        if i < len(book['asks']):
            row[f"ask_{i}_price"] = book['asks'][i].price
            row[f"ask_{i}_size"] = book['asks'][i].size
        else:
            row[f"ask_{i}_price"], row[f"ask_{i}_size"] = 0, 0

    # 计算 Imbalance (本地订单簿)
    bid_vol = sum([b.size for b in book['bids'][:DEPTH]])
    ask_vol = sum([a.size for a in book['asks'][:DEPTH]])
    row["imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

    return row

try:
    print("Collecting LOCAL data... Press Ctrl+C to stop.")
    while True:
        # 修正点 2: 将 GLOBAL 修改为 LOCAL
        book_obj_bid = trader.get_order_book(TICKER, shift.OrderBookType.LOCAL_BID, max_level=DEPTH)
        book_obj_asks = trader.get_order_book(TICKER, shift.OrderBookType.LOCAL_ASK, max_level=DEPTH)
        
        book = {"bids": book_obj_bid, "asks": book_obj_asks}

        row = extract_features(book)
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        data.append(row)
        print(f"{row['timestamp'][-12:]} | mid={row['mid']:.2f} | imb={row['imbalance']:.2f}")

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