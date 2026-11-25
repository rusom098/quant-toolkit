import matplotlib.pyplot as plt
from data import get_prices, get_equity, get_max_drawdown, get_annual_return, get_volatility, get_sharpe

def ma_crossover_signal(prices, short=50, long=200):
    short_ma = prices.rolling(short).mean()
    long_ma = prices.rolling(long).mean()
    signal = (short_ma > long_ma).astype(int)
    return signal

def run_backtest(prices, signal, cost=0.001):
    returns = prices.pct_change().fillna(0)
    position = signal.shift(1).fillna(0)
    trades = position.diff().abs().fillna(0)
    strategy_returns = position * returns -trades * cost
    return strategy_returns


def print_stats(name, returns):
    equity = get_equity(returns)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    print(f"{name:<12} Total: {total:>8.1%} Annual: {get_annual_return(equity):>6.1%}"
          f"Volatility: {get_volatility(returns):>6.1%}  Sharpe: {get_sharpe(returns):5.2f}  "
          f"Max DD: {get_max_drawdown(equity):>7.1%}")
    return equity



ticker = input("Enter a ticker: ")
prices = get_prices(ticker, "2015-01-01", "2025-01-01")

signal = ma_crossover_signal(prices)
strategy = run_backtest(prices, signal)
buy_hold = prices.pct_change().fillna(0)

print(f"\n{ticker}: 50/200-day moving average crossover vs buy and hold")
strat_equity = print_stats("Strategy", strategy)
bh_equity = print_stats("Buy & Hold", buy_hold)
print(f"Number of trades: {int(signal.shift(1).fillna(0).diff().abs().sum())}")

plt.style.use("dark_background")
plt.figure(figsize=(12, 6), facecolor="#262624")
plt.gca().set_facecolor("#262624")
plt.plot(strat_equity, color="#A8C5DA", label="MA Crossover")
plt.plot(bh_equity, color="#E8B4A0", label="Buy & Hold")
plt.title(f"{ticker}: Strategy vs Buy & Hold", color="#C2C0B6")
plt.ylabel("Portfolio Value ($)", color="#C2C0B6")
plt.legend()
plt.grid(True, alpha=0.1)
plt.show()