import shift
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, time as dt_time
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================
# 🧠 DeepLOB 模型（论文原版轻量化工程版）
# ==============================================
class DeepLOB(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积提取订单簿价+量空间特征
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,10))

        # Inception 模块（多尺度时序）
        self.incep1 = nn.Conv2d(64, 32, kernel_size=(3,1))
        self.incep2 = nn.Conv2d(64, 32, kernel_size=(10,1))
        self.incep3 = nn.Conv2d(64, 32, kernel_size=(20,1))

        # LSTM 捕捉时序趋势
        self.lstm = nn.LSTM(96, 64, batch_first=True, num_layers=2)
        self.fc = nn.Linear(64, 3)  # 3分类：涨/平/跌
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (1, 1, 100, 40) → 100个时间步 × 40个订单簿特征
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # 多尺度时序
        x1 = self.relu(self.incep1(x))
        x2 = self.relu(self.incep2(x))
        x3 = self.relu(self.incep3(x))

        # 拼接
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.squeeze(-1).transpose(1,2)

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# 全局模型（加载一次即可）
DEVICE = torch.device("cpu")
model = DeepLOB().to(DEVICE)
model.eval()

# ==============================================
# 订单簿序列缓存（100步 × 40维 符合DeepLOB输入）
# ==============================================
LOB_SEQ_LENGTH = 100
LOB_FEATURE_DIM = 40  # 10档买卖 × 价+量 × 2

# ==============================================
# 你原来的工具函数（完全保留）
# ==============================================
def cancel_orders(trader: shift.Trader, ticker: str):
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            trader.submit_cancellation(order)
    sleep(0.1)

def close_positions(trader: shift.Trader, ticker: str):
    item = trader.get_portfolio_item(ticker)
    long_shares  = item.get_long_shares()
    short_shares = item.get_short_shares()
    if long_shares > 0:
        lots = long_shares // 100
        if lots > 0:
            o = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(lots))
            trader.submit_order(o)
    if short_shares > 0:
        lots = short_shares // 100
        if lots > 0:
            o = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(lots))
            trader.submit_order(o)
    sleep(1)

def full_cleanup(trader, tickers):
    print("CLEANING...", flush=True)
    for order in list(trader.get_waiting_list()):
        trader.submit_cancellation(order)
    sleep(1)
    for ticker in tickers:
        item = trader.get_portfolio_item(ticker)
        long_shares = item.get_long_shares()
        if long_shares > 0:
            lots = long_shares // 100
            if lots > 0:
                o = shift.Order(shift.Order.Type.MARKET_SELL, ticker, lots)
                trader.submit_order(o)
        short_shares = item.get_short_shares()
        if short_shares > 0:
            lots = short_shares // 100
            if lots > 0:
                o = shift.Order(shift.Order.Type.MARKET_BUY, ticker, lots)
                trader.submit_order(o)
    sleep(2)
    print("CLEAN DONE", flush=True)

# ==============================================
# 🎯 核心：DeepLOB 预测价格方向（替代原来的FFT策略）
# ==============================================
def predict_by_deeplob(lob_history):
    if len(lob_history) < LOB_SEQ_LENGTH:
        return 1  # 数据不足默认中性

    seq = np.array(lob_history[-LOB_SEQ_LENGTH:], dtype=np.float32)
    seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)  # 归一化

    tensor = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,100,40)

    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax(1).item()

    # 0=跌 1=平 2=涨
    return pred

# ==============================================
# 构建订单簿40维特征（买卖各10档 价+量）
# ==============================================
def build_lob_feature(bp):
    feat = []
    # 买1-10
    for i in range(10):
        feat.append(bp.get_bid_price(i) if i <10 else 0)
        feat.append(bp.get_bid_size(i) if i <10 else 0)
    # 卖1-10
    for i in range(10):
        feat.append(bp.get_ask_price(i) if i <10 else 0)
        feat.append(bp.get_ask_size(i) if i <10 else 0)
    return feat

# ==============================================
# 策略主逻辑（替换成DeepLOB）
# ==============================================
def strategy_step(trader: shift.Trader, ticker: str, endtime):
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    check_freq = 1
    base_size  = 1
    max_inv    = 5

    # 缓存100步订单簿（给DeepLOB用）
    lob_history: list[list[float]] = []

    while trader.get_last_trade_time() < endtime:
        cancel_orders(trader, ticker)
        bp = trader.get_best_price(ticker)
        best_bid = bp.get_bid_price()
        best_ask = bp.get_ask_price()

        if best_bid <=0 or best_ask <=0 or best_ask <= best_bid:
            sleep(check_freq)
            continue

        # 构建40维LOB特征
        feat = build_lob_feature(bp)
        lob_history.append(feat)
        if len(lob_history) > LOB_SEQ_LENGTH:
            lob_history.pop(0)

        mid = (best_bid + best_ask)/2
        spread = best_ask - best_bid

        # =======================
        # 🚀 DeepLOB 预测方向
        # =======================
        pred = predict_by_deeplob(lob_history)

        # 仓位
        item = trader.get_portfolio_item(ticker)
        net = (item.get_long_shares() - item.get_short_shares()) // 100

        buy_size  = 0
        sell_size = 0
        bid_px = best_bid
        ask_px = best_ask

        # =======================
        # AI 信号交易逻辑
        # =======================
        if pred == 2:  # 模型预测涨
            buy_size = base_size +1
            bid_px = best_bid
            if net < max_inv:
                buy_size = base_size +1

        elif pred ==0: # 模型预测跌
            sell_size = base_size +1
            ask_px = best_ask
            if net > -max_inv:
                sell_size = base_size +1

        else:
            buy_size = base_size if net < max_inv else 0
            sell_size = base_size if net > -max_inv else 0

        # 挂单
        if buy_size>0:
            bo = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, buy_size)
            bo.price = round(bid_px,2)
            trader.submit_order(bo)
            print(f"[DEEPLOB BUY] {ticker} px={bid_px:.2f} sz={buy_size}")

        if sell_size>0:
            so = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, sell_size)
            so.price = round(ask_px,2)
            trader.submit_order(so)
            print(f"[DEEPLOB SELL] {ticker} px={ask_px:.2f} sz={sell_size}")

        sleep(check_freq)

    cancel_orders(trader, ticker)
    close_positions(trader, ticker)
    print(f"[PnL] {ticker}: {trader.get_portfolio_item(ticker).get_realized_pl()-initial_pl:+.2f}")

# ==============================================
# main 函数（完全保留你的结构）
# ==============================================
def main(trader: shift.Trader):
    current = trader.get_last_trade_time()
    end_time = datetime.combine(current.date(), dt_time(15,50,0))
    while trader.get_last_trade_time() < current:
        sleep(1)

    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
    tickers = ["AAPL","MSFT","V"]
    print("START", flush=True)

    with ThreadPoolExecutor(len(tickers)) as pool:
        futures = {pool.submit(strategy_step, trader,t,end_time):t for t in tickers}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"ERROR {futures[f]}: {e}")

    print("END")
    print(f"FINAL BP: {trader.get_portfolio_summary().get_total_bp():.2f}")
    print(f"FINAL PnL: {trader.get_portfolio_summary().get_total_realized_pl()-initial_pl:+.2f}")

if __name__ == "__main__":
    with shift.Trader("four-sigma") as trader:
        trader.connect("initiator.cfg", "sk8Pf6nJ")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)
        full_cleanup(trader, ["AAPL","MSFT","V"])
        main(trader)