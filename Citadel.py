import shift
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, time as dt_time, timedelta
from time import sleep
from typing import Deque, Dict, List, Optional, Tuple


TICKERS = ["CSCO","MSFT","NVDA"]

# Session timing (Adjust these to fit your active SHIFT trading hours)
END_TIME = dt_time(15, 50, 0)
SOFT_FLATTEN_TIME = dt_time(15, 47, 0)
HARD_FLATTEN_TIME = dt_time(15, 49, 20)

# Loop speed
SLEEP_SEC = 0.25
CANCEL_PAUSE_SEC = 0.05

# Trading / risk
LOTS_PER_ORDER_MIN = 1
LOTS_PER_ORDER_MAX = 2
BASE_MAX_INVENTORY = 4        # per ticker, in lots
BASE_MAX_GROSS_INVENTORY = 4   # across tickers, in lots
MIN_BP_BUFFER = 25_000          # avoid using the last bit of buying power

# Quote management
QUOTE_MAX_AGE_SEC = 1.2
REPRICE_THRESHOLD = 0.01        # dollars
IMPROVE_ONE_TICK_IF_POSSIBLE = True

# Volatility model
EWMA_LAMBDA = 0.94
VOL_FLOOR = 1e-6

# Online regression model
FEATURE_WINDOW = 180
RIDGE_LAMBDA = 1e-3
MIN_TRAIN_SAMPLES = 30

# Alpha / quoting thresholds
EDGE_TO_QUOTE = 0.0060          # predicted return threshold in dollars
EDGE_TO_ONE_SIDE = 0.012
EDGE_TO_TAKE = 0.0250

# Spread model
BASE_HALF_SPREAD = 0.015
VOL_HALF_SPREAD_MULT = 3.0
INVENTORY_HALF_SPREAD_MULT = 0.004
ADVERSE_SELECTION_MULT = 0.25

# Reservation price skew
INVENTORY_SKEW = 0.02
LATE_DAY_INVENTORY_SKEW = 0.04

# Expected PnL filter
FILL_DECAY = 80.0               # higher -> more penalty for further quotes
FIXED_COST_PER_SHARE = 0.0005   # rough penalty proxy
VOL_COST_MULT = 0.30
INVENTORY_COST_MULT = 0.0015

# Liquidity / regime filters
MIN_OBSERVED_SPREAD = 0.02
MAX_OBSERVED_SPREAD = 0.15
MIN_DEPTH = 1
MAX_VOL_PRICE = 0.06

# Market order cooldown
TAKER_COOLDOWN_SEC = 8.0

# Debug
PRINT_DEBUG = True


@dataclass
class PendingLabel:
    x: np.ndarray
    mid: float


@dataclass
class SymbolState:
    prev_mid: Optional[float] = None
    ewma_var: float = VOL_FLOOR
    imbalance_ema: float = 0.0
    fast_ret_ema: float = 0.0
    slow_ret_ema: float = 0.0
    X_hist: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=FEATURE_WINDOW))
    y_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=FEATURE_WINDOW))
    beta: Optional[np.ndarray] = None
    pending: Optional[PendingLabel] = None
    last_bid_quote: Optional[float] = None
    last_ask_quote: Optional[float] = None
    last_quote_time: Optional[datetime] = None
    had_quotes: bool = False
    last_taker_time: Optional[datetime] = None
    samples: int = 0


def debug(msg: str):
    if PRINT_DEBUG:
        print(msg, flush=True)


def round_down_cent(px: float) -> float:
    return round(np.floor(px * 100.0) / 100.0, 2)


def round_up_cent(px: float) -> float:
    return round(np.ceil(px * 100.0) / 100.0, 2)


def seconds_since(now: datetime, old: Optional[datetime]) -> float:
    if old is None:
        return 1e18
    return (now - old).total_seconds()


def safe_log_return(new_px: float, old_px: Optional[float]) -> float:
    if old_px is None or old_px <= 0 or new_px <= 0:
        return 0.0
    return float(np.log(new_px / old_px))


def microprice(best_bid: float, best_ask: float, bid_size: int, ask_size: int) -> float:
    total = bid_size + ask_size
    if total <= 0:
        return 0.5 * (best_bid + best_ask)
    return (best_bid * ask_size + best_ask * bid_size) / total


def imbalance(bid_size: int, ask_size: int) -> float:
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


def get_inventory_lots(trader: shift.Trader, ticker: str) -> int:
    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()
    return int((long_shares - short_shares) / 100)


def get_gross_inventory_lots(trader: shift.Trader, tickers: List[str]) -> int:
    return int(sum(abs(get_inventory_lots(trader, t)) for t in tickers))


def get_bp(trader: shift.Trader) -> float:
    return float(trader.get_portfolio_summary().get_total_bp())


def dynamic_inventory_limit(now: datetime, current_day: datetime.date) -> int:
    soft = datetime.combine(current_day, SOFT_FLATTEN_TIME)
    hard = datetime.combine(current_day, HARD_FLATTEN_TIME)
    if now < soft:
        return BASE_MAX_INVENTORY
    total = max((hard - soft).total_seconds(), 1.0)
    left = max((hard - now).total_seconds(), 0.0)
    scaled = int(np.ceil(BASE_MAX_INVENTORY * left / total))
    return max(1, scaled)


def cancel_orders_for_ticker(trader: shift.Trader, ticker: str):
    for order in list(trader.get_waiting_list()):
        if order.symbol == ticker:
            trader.submit_cancellation(order)
    sleep(CANCEL_PAUSE_SEC)


def clear_quotes(trader: shift.Trader, ticker: str, state: SymbolState):
    if state.had_quotes:
        cancel_orders_for_ticker(trader, ticker)
    state.last_bid_quote = None
    state.last_ask_quote = None
    state.last_quote_time = None
    state.had_quotes = False


def close_positions(trader: shift.Trader, ticker: str):
    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()
    if long_shares > 0:
        lots = long_shares // 100
        if lots > 0:
            order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(lots))
            trader.submit_order(order)
    if short_shares > 0:
        lots = short_shares // 100
        if lots > 0:
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(lots))
            trader.submit_order(order)


def full_cleanup(trader: shift.Trader, tickers: List[str]):
    debug("CLEANING...")
    for order in list(trader.get_waiting_list()):
        trader.submit_cancellation(order)
    sleep(0.5)
    for ticker in tickers:
        close_positions(trader, ticker)
    sleep(1.0)
    debug("CLEAN DONE")


def build_feature_vector(mid: float, micro: float, spread: float, imb_now: float, 
                         imb_ema: float, fast_ret_ema: float, slow_ret_ema: float, 
                         vol_price: float, inventory_lots: int) -> np.ndarray:
    micro_edge = micro - mid
    trend = fast_ret_ema - slow_ret_ema
    return np.array([1.0, micro_edge, spread, imb_now, imb_ema, trend, vol_price, float(inventory_lots)], dtype=float)


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    n_features = X.shape[1]
    reg = ridge_lambda * np.eye(n_features)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return beta


def maybe_update_model(state: SymbolState):
    if len(state.X_hist) < MIN_TRAIN_SAMPLES:
        return
    X = np.vstack(state.X_hist)
    y = np.array(state.y_hist, dtype=float)
    try:
        state.beta = fit_ridge(X, y, RIDGE_LAMBDA)
    except np.linalg.LinAlgError:
        state.beta, *_ = np.linalg.lstsq(X, y, rcond=None)


def predict_next_return(state: SymbolState, x: np.ndarray) -> float:
    if state.beta is None:
        return (0.80 * x[1] + 0.0020 * x[3] + 0.0030 * x[4] + 5.0 * x[5] - 0.20 * x[6] - 0.0008 * x[7] + 0.02 * (x[2] - 0.02))
    return float(state.beta @ x)


def should_requote(now: datetime, state: SymbolState, new_bid: Optional[float], new_ask: Optional[float]) -> bool:
    if not state.had_quotes:
        return True
    if seconds_since(now, state.last_quote_time) >= QUOTE_MAX_AGE_SEC:
        return True
    if (state.last_bid_quote is None) != (new_bid is None):
        return True
    if (state.last_ask_quote is None) != (new_ask is None):
        return True
    moved = False
    if state.last_bid_quote is not None and new_bid is not None:
        moved |= abs(state.last_bid_quote - new_bid) >= REPRICE_THRESHOLD
    if state.last_ask_quote is not None and new_ask is not None:
        moved |= abs(state.last_ask_quote - new_ask) >= REPRICE_THRESHOLD
    return moved


def compute_quote_sizes(predicted_ret: float, vol_price: float, inventory_lots: int, 
                        inv_limit: int, observed_spread: float, now: datetime, session_day: datetime.date) -> Tuple[int, int]:
    size = LOTS_PER_ORDER_MIN
    if abs(predicted_ret) >= EDGE_TO_ONE_SIDE:
        size += 1
    if observed_spread >= 0.03:
        size += 1
    if vol_price >= 0.03:
        size -= 1
    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        size = 1
    size = int(np.clip(size, LOTS_PER_ORDER_MIN, LOTS_PER_ORDER_MAX))
    buy_size = size
    sell_size = size
    if predicted_ret > 0:
        buy_size = min(LOTS_PER_ORDER_MAX, buy_size + 1)
        sell_size = max(0, sell_size - 1)
    elif predicted_ret < 0:
        sell_size = min(LOTS_PER_ORDER_MAX, sell_size + 1)
        buy_size = max(0, buy_size - 1)
    if inventory_lots >= inv_limit:
        buy_size = 0
        sell_size = max(1, sell_size)
    elif inventory_lots <= -inv_limit:
        sell_size = 0
        buy_size = max(1, buy_size)
    return buy_size, sell_size


def expected_fill_prob(distance_from_touch: float) -> float:
    distance = max(distance_from_touch, 0.0)
    return float(np.exp(-FILL_DECAY * distance))


def expected_trade_value(quoted_edge_per_share: float, fill_prob: float, vol_price: float, inventory_lots: int) -> float:
    cost = (FIXED_COST_PER_SHARE + VOL_COST_MULT * vol_price + INVENTORY_COST_MULT * abs(inventory_lots))
    return fill_prob * quoted_edge_per_share - cost


def compute_quotes(best_bid: float, best_ask: float, fair_price: float, observed_spread: float, 
                   vol_price: float, predicted_ret: float, inventory_lots: int, now: datetime, 
                   session_day: datetime.date) -> Tuple[Optional[float], Optional[float]]:
    half_spread = max(BASE_HALF_SPREAD, 0.5 * observed_spread, VOL_HALF_SPREAD_MULT * vol_price, 
                      INVENTORY_HALF_SPREAD_MULT * abs(inventory_lots), ADVERSE_SELECTION_MULT * abs(predicted_ret))
    reservation = fair_price - INVENTORY_SKEW * inventory_lots
    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        reservation -= LATE_DAY_INVENTORY_SKEW * inventory_lots

    raw_bid = reservation - half_spread
    raw_ask = reservation + half_spread

    if IMPROVE_ONE_TICK_IF_POSSIBLE and observed_spread >= 0.02:
        bid_quote = min(best_bid + 0.01, round_down_cent(raw_bid))
        ask_quote = max(best_ask - 0.01, round_up_cent(raw_ask))
    else:
        bid_quote = min(best_bid, round_down_cent(raw_bid))
        ask_quote = max(best_ask, round_up_cent(raw_ask))

    bid_quote = min(bid_quote, round(best_ask - 0.01, 2))
    ask_quote = max(ask_quote, round(best_bid + 0.01, 2))

    if bid_quote >= ask_quote:
        bid_quote = round(best_bid, 2)
        ask_quote = round(best_ask, 2)

    bid_enabled = True
    ask_enabled = True

    if predicted_ret >= EDGE_TO_ONE_SIDE and inventory_lots <= 0:
        ask_enabled = False
    elif predicted_ret <= -EDGE_TO_ONE_SIDE and inventory_lots >= 0:
        bid_enabled = False

    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        if inventory_lots > 0:
            bid_enabled = False
            ask_quote = round(best_ask, 2)
        elif inventory_lots < 0:
            ask_enabled = False
            bid_quote = round(best_bid, 2)

    return bid_quote if bid_enabled else None, ask_quote if ask_enabled else None


def process_new_observation(state: SymbolState, mid: float, x_now: np.ndarray):
    if state.pending is not None:
        y = mid - state.pending.mid
        state.X_hist.append(state.pending.x)
        state.y_hist.append(y)
        maybe_update_model(state)
    state.pending = PendingLabel(x=x_now, mid=mid)


def step_symbol(trader: shift.Trader, ticker: str, tickers: List[str], state: SymbolState, now: datetime, session_day: datetime.date):
    best = trader.get_best_price(ticker)
    best_bid = best.get_bid_price()
    best_ask = best.get_ask_price()
    bid_size = best.get_bid_size()
    ask_size = best.get_ask_size()

    # ADDED DEBUGS TO SEE WHAT THE API IS GIVING YOU
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        # Avoid flood logging, just print once in a while if data is missing
        if state.samples % 20 == 0:
            debug(f"[SKIP] {ticker} invalid book. Bid: {best_bid}, Ask: {best_ask}")
        clear_quotes(trader, ticker, state)
        return

    spread = round(best_ask - best_bid, 2)
    if spread < MIN_OBSERVED_SPREAD or spread > MAX_OBSERVED_SPREAD:
        clear_quotes(trader, ticker, state)
        return

    mid = 0.5 * (best_bid + best_ask)
    micro = microprice(best_bid, best_ask, bid_size, ask_size)
    imb_now = imbalance(bid_size, ask_size)
    ret = safe_log_return(mid, state.prev_mid)
    
    state.ewma_var = EWMA_LAMBDA * state.ewma_var + (1.0 - EWMA_LAMBDA) * (ret ** 2)
    vol_price = mid * np.sqrt(max(state.ewma_var, VOL_FLOOR))
    state.imbalance_ema = 0.25 * imb_now + 0.75 * state.imbalance_ema
    state.fast_ret_ema = 0.30 * ret + 0.70 * state.fast_ret_ema
    state.slow_ret_ema = 0.08 * ret + 0.92 * state.slow_ret_ema
    state.samples += 1

    inventory_lots = get_inventory_lots(trader, ticker)
    gross_inventory_lots = get_gross_inventory_lots(trader, tickers)
    inv_limit = dynamic_inventory_limit(now, session_day)

    x_now = build_feature_vector(mid, micro, spread, imb_now, state.imbalance_ema, state.fast_ret_ema, state.slow_ret_ema, vol_price, inventory_lots)
    process_new_observation(state, mid, x_now)
    predicted_ret = predict_next_return(state, x_now)

    if state.samples < 12:
        state.prev_mid = mid
        clear_quotes(trader, ticker, state)
        return

    if abs(predicted_ret) < EDGE_TO_QUOTE and spread <= 0.02:
        clear_quotes(trader, ticker, state)
        state.prev_mid = mid
        return

    fair_price = micro + predicted_ret
    bid_quote, ask_quote = compute_quotes(best_bid, best_ask, fair_price, spread, vol_price, predicted_ret, inventory_lots, now, session_day)
    buy_size, sell_size = compute_quote_sizes(predicted_ret, vol_price, inventory_lots, inv_limit, spread, now, session_day)

    if should_requote(now, state, bid_quote, ask_quote):
        clear_quotes(trader, ticker, state)

        # ADDED STRICT 2-DECIMAL ROUNDING FOR SHIFT ACCEPTANCE
        if bid_quote is not None and buy_size > 0:
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, int(buy_size))
            order.price = round(float(bid_quote), 2)
            trader.submit_order(order)
            debug(f"[BID] {ticker} px={bid_quote:.2f} sz={buy_size}")

        if ask_quote is not None and sell_size > 0:
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, int(sell_size))
            order.price = round(float(ask_quote), 2)
            trader.submit_order(order)
            debug(f"[ASK] {ticker} px={ask_quote:.2f} sz={sell_size}")

        state.last_bid_quote = bid_quote
        state.last_ask_quote = ask_quote
        state.last_quote_time = now
        state.had_quotes = True

    state.prev_mid = mid


def main(trader: shift.Trader):
    # CRITICAL FIX: We are using local machine clock now so simulation 
    # server time anomalies won't instantly break/end the loop.
    session_day = datetime.now().date()
    end_dt = datetime.combine(session_day, END_TIME)
    hard_flatten_dt = datetime.combine(session_day, HARD_FLATTEN_TIME)

    states: Dict[str, SymbolState] = {ticker: SymbolState() for ticker in TICKERS}
    initial_realized = trader.get_portfolio_summary().get_total_realized_pl()

    debug("START")
    debug(f"Trading loop active until {end_dt.time()}")

    while datetime.now() < end_dt:
        now = datetime.now()

        if now >= hard_flatten_dt:
            debug("[FLATTEN] hard flatten window reached")
            full_cleanup(trader, TICKERS)
            break

        for ticker in TICKERS:
            try:
                step_symbol(trader, ticker, TICKERS, states[ticker], now, session_day)
            except Exception as exc:
                debug(f"[ERROR] {ticker}: {exc}")

        sleep(SLEEP_SEC)

    full_cleanup(trader, TICKERS)
    final_realized = trader.get_portfolio_summary().get_total_realized_pl()
    debug("END")
    debug(f"final PnL: {final_realized - initial_realized:+.2f}")


if __name__ == "__main__":
    with shift.Trader("four-sigma") as trader:
        trader.connect("initiator.cfg", "sk8Pf6nJ")
        sleep(2)

        trader.sub_all_order_book()
        sleep(5)  # Increased wait time for initial stream population

        # DIAGNOSTIC TEST: Check if any data exists at all
        test_ticker = "AAPL"
        test_best = trader.get_best_price(test_ticker)
        print(f"DIAGNOSTIC - Best Bid for {test_ticker}: {test_best.get_bid_price()}")

        if test_best.get_bid_price() == 0.0:
            print("WARNING: The environment is returning no data. Wait until market hours or check ticker permissions.")
        
        full_cleanup(trader, TICKERS)
        main(trader)