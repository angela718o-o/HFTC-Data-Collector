
import os
# 彻底屏蔽 PyTorch NNPACK 的硬件不支持警告，阻止其刷屏
os.environ['MKLDNN_VERBOSE'] = '0'
os.environ['NNPACK_VERBOSE'] = '0'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR' 

import shift
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, time as dt_time
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 🧠 2. DeepLOB 模型（包含 Inception 与 BatchNorm2d）
# ==============================================================================
class DeepLOB(nn.Module):
    def __init__(self):
        super().__init__()
        # 全网使用 LeakyReLU(0.01)
        lrelu_neg_slope = 0.01

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 10))

        # 分支 1: 1x1 卷积 -> 3x1 卷积
        self.incep1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )
        
        # 分支 2: 1x1 卷积 -> 5x1 卷积
        self.incep2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )
        
        # 分支 3: 3x1 MaxPool -> 1x1 卷积
        self.incep3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )

        self.lstm = nn.LSTM(96, 64, batch_first=True, num_layers=1)
        self.fc = nn.Linear(64, 3)
        self.relu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x1 = self.incep1(x)
        x2 = self.incep2(x)
        x3 = self.incep3(x)

        x = torch.cat([x1, x2, x3], dim=1) 
        x = x.squeeze(-1).transpose(1, 2)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# ==============================================================================
# 全局模型初始化
# ==============================================================================
DEVICE = torch.device("cpu")
model = DeepLOB().to(DEVICE)
# 加载之前训练好的权重
try:
    model.load_state_dict(torch.load("deeplob_weight.pth"))
    print("✅ 成功加载 deeplob_weight.pth")
except Exception as e:
    print(f"⚠️ 权重加载失败，请检查文件是否存在: {e}")
model.eval()

LOB_SEQ_LENGTH = 100

# ==============================================================================
# 🛠️ 3. 辅助工具函数
# ==============================================================================
def cancel_orders(trader: shift.Trader, ticker: str):
    waiting = trader.get_waiting_list()
    if not waiting:
        return
    for order in waiting:
        if order.symbol == ticker:
            trader.submit_cancellation(order)

def close_positions(trader: shift.Trader, ticker: str):
    item = trader.get_portfolio_item(ticker)
    long_shares  = item.get_long_shares()
    short_shares = item.get_short_shares()
    if long_shares > 0:
        lots = long_shares // 100
        if lots > 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(lots)))
    if short_shares > 0:
        lots = short_shares // 100
        if lots > 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(lots)))
    sleep(1)

def full_cleanup(trader, tickers):
    print("🧹 [CLEANUP] 正在撤单并平仓...", flush=True)
    for order in list(trader.get_waiting_list()):
        trader.submit_cancellation(order)
    sleep(1)
    for ticker in tickers:
        close_positions(trader, ticker)
    print("🧹 [CLEANUP] 清理完成", flush=True)

# ==============================================================================
# 🧠 4. 预测函数 (极致低延迟改造)
# ==============================================================================
def predict_deeplob(lob_tensor):
    """直接接收外部已经准备好的 Tensor，使用 inference_mode 排除多余计算"""
    with torch.inference_mode():
        out = model(lob_tensor)
        return out.argmax(1).item()

# ==============================================================================
# 📊 5. 策略主逻辑
# ==============================================================================
def strategy_step(trader: shift.Trader, ticker: str, endtime):
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    base_size = 1
    max_inv = 5

    # ⚡ 预分配 NumPy 缓冲区和 Torch Tensor，拒绝在循环中频繁 malloc 内存
    lob_history = np.zeros((LOB_SEQ_LENGTH, 40), dtype=np.float32)
    seq_tensor = torch.zeros((1, 1, LOB_SEQ_LENGTH, 40), dtype=torch.float32).to(DEVICE)
    
    ptr = 0
    filled_len = 0
    last_lob_state = None

    print(f"📡 {ticker} 策略线程已启动...")

    while trader.get_last_trade_time() < endtime:
        bp = trader.get_best_price(ticker)
        best_bid = bp.get_bid_price()
        best_ask = bp.get_ask_price()

        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            sleep(0.001)  # 防止 CPU 空转
            continue

        # 构建当前盘口特征（由于 API 限制只有 L1，我们复制 10 次填满 40 维以匹配模型输入）
        current_feat = [best_bid, bp.get_bid_size(), best_ask, bp.get_ask_size()] * 10
        
        # ⚡ 核心修复 1：如果盘口完全没有发生任何跳变，跳过本次预测，避免 AI 面对死寂数据饱和
        if last_lob_state is not None and np.allclose(current_feat, last_lob_state):
            sleep(0.005) # 稍微休眠，等待下一次 tick
            continue
            
        last_lob_state = current_feat  # 更新状态记录

        # 环形缓冲区写入 (避免 List pop(0) 带来的全量内存搬移)
        lob_history[ptr] = current_feat
        ptr = (ptr + 1) % LOB_SEQ_LENGTH
        if filled_len < LOB_SEQ_LENGTH:
            filled_len += 1

        # 如果历史数据还没攒满 100 条，先不预测
        if filled_len < LOB_SEQ_LENGTH:
            sleep(0.001)
            continue

        # 整理出按时间先后排序的历史切片
        if ptr == 0:
            ordered_seq = lob_history
        else:
            ordered_seq = np.concatenate([lob_history[ptr:], lob_history[:ptr]], axis=0)

        # ⚡ 核心修复 2：防止静态数据导致 Z-score 标准差为 0 引起的除以零异常放大
        data_std = ordered_seq.std(axis=0)
        if np.mean(data_std) < 1e-5:
            # 如果盘口毫无波动，直接把时序数据清零，避免 AI 胡乱输出
            norm_seq = np.zeros_like(ordered_seq)
        else:
            norm_seq = (ordered_seq - ordered_seq.mean(axis=0)) / (data_std + 1e-8)

        # 零拷贝覆盖到预分配的 Tensor 中
        seq_tensor[0, 0, :, :] = torch.from_numpy(norm_seq)

        # 🧠 AI 预测 (0: 下跌, 1: 走平, 2: 上涨)
        pred = predict_deeplob(seq_tensor)

        # 获取当前仓位
        item = trader.get_portfolio_item(ticker)
        net = (item.get_long_shares() - item.get_short_shares()) // 100

        buy_size = 0
        sell_size = 0

        if pred == 2:
            buy_size = base_size + 1 if net < max_inv else 0
        elif pred == 0:
            sell_size = base_size + 1 if net > -max_inv else 0
        else:
            # 走平时稍微降低交易激进性，用于回撤防守
            buy_size = base_size if net < max_inv else 0
            sell_size = base_size if net > -max_inv else 0

        # ⚡ 核心优化：只有在真正需要下单时，才去触发撤单 I/O！避免无意义的通信开销
        if buy_size > 0 or sell_size > 0:
            cancel_orders(trader, ticker)

        if buy_size > 0:
            bo = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, buy_size)
            bo.price = round(best_bid, 2)
            trader.submit_order(bo)
            print(f"🟢 [DEEPLOB BUY] {ticker} 价格:{best_bid:.2f} 数量:{buy_size}")

        if sell_size > 0:
            so = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, sell_size)
            so.price = round(best_ask, 2)
            trader.submit_order(so)
            print(f"🔴 [DEEPLOB SELL] {ticker} 价格:{best_ask:.2f} 数量:{sell_size}")

        # 核心休眠：改为 10 毫秒一次轮询（每秒检查 100 次最新盘口变化）
        sleep(0.01)

    # 交易时间结束，清理该标的的残余仓位
    cancel_orders(trader, ticker)
    close_positions(trader, ticker)
    print(f"🏁 [PnL] {ticker}: {trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl:+.2f}")

# ==============================================================================
# 🚪 6. 主程序入口
# ==============================================================================
def main(trader: shift.Trader):
    current = trader.get_last_trade_time()
    end_time = datetime.combine(current.date(), dt_time(15, 50, 0))
    
    while trader.get_last_trade_time() < current:
        sleep(1)

    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
    tickers = ["AAPL"]
    print("🔥 策略正式启动 🔥", flush=True)

    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        futures = {pool.submit(strategy_step, trader, t, end_time): t for t in tickers}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"❌ [ERROR] 标的 {futures[fut]} 发生异常: {exc}", flush=True)

    print("🏁 交易时间结束 🏁")
    print(f"最终购买力: {trader.get_portfolio_summary().get_total_bp():.2f}")
    print(f"总实现盈亏: {trader.get_portfolio_summary().get_total_realized_pl() - initial_pl:+.2f}")

if __name__ == "__main__":
    with shift.Trader("four-sigma") as trader:
        trader.connect("initiator.cfg", "sk8Pf6nJ")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)
        
        # 启动前全面清理一次
        full_cleanup(trader, ["AAPL"])
        
        # 运行主程序
        main(trader)