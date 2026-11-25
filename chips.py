import matplotlib.pyplot as plt
from data import get_prices, get_returns, get_summary

CHIP_TICKERS = ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU",
                "TSM", "ASML", "AMAT", "LRCX", "KLAC"]
BENCHMARK = "SMH"  # VanEck Semiconductor ETF, the sector benchmark
START = "2015-01-01"
END = "2025-01-01"


# Download one ticker and work out its metrics. Returns None if no data.
def get_stock_metrics(ticker, start=START, end=END):
    prices = get_prices(ticker, start, end).dropna()
    if prices.empty:  # download failed or rate-limited
        print(f"{ticker}: no data downloaded (possible yfinance rate limit)")
        return None

    returns = get_returns(prices)
    metrics = get_summary(returns)  # total, annual, volatility, sharpe, max_dd
    metrics["ticker"] = ticker
    return metrics


# Sort stocks by Sharpe ratio, best first.
def sort_by_sharpe(rows):
    return sorted(rows, key=lambda row: row["sharpe"], reverse=True)


# Did the stock beat the benchmark's annual return?
def get_verdict(row, benchmark_row):
    if row is benchmark_row:
        return "(benchmark)"
    if row["annual"] > benchmark_row["annual"]:
        return "beat"
    return "lagged"


# Print a table of all stocks, sorted by Sharpe ratio (best first).
def print_dashboard(rows, benchmark_row):
    rows = sort_by_sharpe(rows)
    print(f"\nChip sector dashboard, {START} to {END} (sorted by Sharpe)")
    print(f"{'Ticker':<8}{'Total':>10}{'Annual':>9}{'Vol':>8}{'Sharpe':>8}{'Max DD':>9}  vs SMH")
    for row in rows + [benchmark_row]:
        if row is benchmark_row:
            print("-" * 62)
        verdict = get_verdict(row, benchmark_row)
        print(f"{row['ticker']:<8}{row['total']:>10.1%}{row['annual']:>9.1%}"
              f"{row['volatility']:>8.1%}{row['sharpe']:>8.2f}{row['max_dd']:>9.1%}  {verdict}")


# Scatter chart: risk (volatility) on x, annual return on y, one dot per stock.
def plot_risk_return(rows, benchmark_row):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#262624")
    ax.set_facecolor("#262624")

    # Individual stocks in blue, each labeled with its ticker
    for row in rows:
        ax.scatter(row["volatility"] * 100, row["annual"] * 100, s=70,
                   color="#A8C5DA", edgecolors="#262624", linewidths=2, zorder=3)
        ax.annotate(row["ticker"], (row["volatility"] * 100, row["annual"] * 100),
                    xytext=(7, 4), textcoords="offset points", color="#C2C0B6", fontsize=10)

    # The benchmark in peach so it stands out
    ax.scatter(benchmark_row["volatility"] * 100, benchmark_row["annual"] * 100, s=110,
               color="#E8B4A0", edgecolors="#262624", linewidths=2, zorder=4)
    ax.annotate("SMH (benchmark)", (benchmark_row["volatility"] * 100, benchmark_row["annual"] * 100),
                xytext=(7, -14), textcoords="offset points", color="#C2C0B6", fontsize=10)  # below the dot, away from TSM

    # Dashed line at SMH's return: dots above it beat the sector
    ax.axhline(benchmark_row["annual"] * 100, color="#E8B4A0", alpha=0.4, linestyle="--", linewidth=1)

    ax.set_title(f"Chip stocks: annual return vs volatility ({START[:4]}-{END[:4]})", color="#C2C0B6")
    ax.set_xlabel("Annualized volatility (%)", color="#C2C0B6")
    ax.set_ylabel("Annual return, CAGR (%)", color="#C2C0B6")
    ax.tick_params(colors="#C2C0B6")
    for spine in ax.spines.values():  # soften the white frame around the chart
        spine.set_color("#5A5955")
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rows = []
    for ticker in CHIP_TICKERS:
        row = get_stock_metrics(ticker)
        if row is not None:
            rows.append(row)
    benchmark_row = get_stock_metrics(BENCHMARK)

    if benchmark_row is None:
        print("Could not download SMH, so there is no benchmark to compare against.")
    else:
        print_dashboard(rows, benchmark_row)
        plot_risk_return(rows, benchmark_row)
