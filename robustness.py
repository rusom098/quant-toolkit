from data import get_prices, get_equity, get_annual_return, get_sharpe, get_max_drawdown
from backtester import compare_strategies

TICKERS = ["QQQ", "SPY", "NVDA", "JPM", "XOM", "AAPL"]
START = "2015-01-01"
END = "2025-01-01"


# Work out the main metrics for one series of daily returns.
def get_metrics(returns):
    equity = get_equity(returns)
    metrics = {
        "annual": get_annual_return(equity),
        "sharpe": get_sharpe(returns),
        "max_dd": get_max_drawdown(equity),
    }
    return metrics


# Backtest one ticker and return a row of results (or None if no data).
def test_ticker(ticker):
    prices = get_prices(ticker, START, END)
    if prices.dropna().empty:  # download failed or rate-limited
        print(f"{ticker}: no data downloaded (possible yfinance rate limit)")
        return None

    strategy, buy_hold, trades = compare_strategies(prices)
    row = {
        "ticker": ticker,
        "start": strategy.index[0].date(),
        "strategy": get_metrics(strategy),
        "buy_hold": get_metrics(buy_hold),
        "trades": trades,
    }
    return row


# Print one results table with strategy and buy-and-hold side by side.
def print_table(rows):
    print(f"\nMA crossover (50/200) vs buy & hold, {START} to {END}, fair start")
    print(f"{'Ticker':<7}{'Start':<12}{'Annual S':>9}{'Annual BH':>10}"
          f"{'Sharpe S':>9}{'Sharpe BH':>10}{'MaxDD S':>9}{'MaxDD BH':>10}{'Trades':>7}")
    for row in rows:
        s = row["strategy"]
        b = row["buy_hold"]
        print(f"{row['ticker']:<7}{str(row['start']):<12}"
              f"{s['annual']:>9.1%}{b['annual']:>10.1%}"
              f"{s['sharpe']:>9.2f}{b['sharpe']:>10.2f}"
              f"{s['max_dd']:>9.1%}{b['max_dd']:>10.1%}{row['trades']:>7}")


# Count how often the strategy beat buy-and-hold on each metric.
def print_summary(rows):
    better_sharpe = 0
    smaller_dd = 0
    higher_return = 0
    for row in rows:
        s = row["strategy"]
        b = row["buy_hold"]
        if s["sharpe"] > b["sharpe"]:
            better_sharpe += 1
        if s["max_dd"] > b["max_dd"]:  # drawdowns are negative, so bigger = smaller drop
            smaller_dd += 1
        if s["annual"] > b["annual"]:
            higher_return += 1
    total = len(rows)
    print(f"\nStrategy had a higher Sharpe on {better_sharpe}/{total} tickers")
    print(f"Strategy had a smaller max drawdown on {smaller_dd}/{total} tickers")
    print(f"Strategy had a higher annual return on {higher_return}/{total} tickers")


if __name__ == "__main__":
    rows = []
    for ticker in TICKERS:
        row = test_ticker(ticker)
        if row is not None:
            rows.append(row)
    print_table(rows)
    print_summary(rows)
