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
def test_ticker(ticker, start=START, end=END):
    prices = get_prices(ticker, start, end)
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
def count_wins(rows):
    wins = {"sharpe": 0, "max_dd": 0, "annual": 0}
    for row in rows:
        s = row["strategy"]
        b = row["buy_hold"]
        if s["sharpe"] > b["sharpe"]:
            wins["sharpe"] += 1
        # Drawdowns are negative, so bigger = smaller drop. They are often exactly
        # equal (both fully invested in the same crash), so a gap under 0.01%
        # counts as a tie instead of a win caused by rounding noise.
        if s["max_dd"] - b["max_dd"] > 0.0001:
            wins["max_dd"] += 1
        if s["annual"] > b["annual"]:
            wins["annual"] += 1
    return wins


# Print the win counts from count_wins.
def print_summary(rows):
    wins = count_wins(rows)
    total = len(rows)
    print(f"\nStrategy had a higher Sharpe on {wins['sharpe']}/{total} tickers")
    print(f"Strategy had a smaller max drawdown on {wins['max_dd']}/{total} tickers")
    print(f"Strategy had a higher annual return on {wins['annual']}/{total} tickers")


if __name__ == "__main__":
    rows = []
    for ticker in TICKERS:
        row = test_ticker(ticker)
        if row is not None:
            rows.append(row)
    print_table(rows)
    print_summary(rows)
