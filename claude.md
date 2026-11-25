# Quantitative Research Toolkit

A Python toolkit for analyzing semiconductor stocks and how they react to
geopolitical events like U.S. export controls. It also includes a general
stock analyzer and a strategy backtester.

This is a learning project. I'm a sophomore in Computer Engineering and a
beginner in both Python and finance.

## How to help me

You write the code. I study it separately, so keep it easy to learn from.

- Build ONE roadmap step at a time. Don't start the next step until I ask.
- Keep changes small. Don't rewrite or reorganize files I didn't ask about.
- Write simple, beginner-readable Python: clear variable names, short
  functions, no clever one-liners or advanced tricks unless necessary.
- Add a short comment above each function saying what it does, and brief
  comments on any line that isn't obvious.
- Reuse the functions in `data.py` instead of duplicating logic.
- After each change, tell me in plain words: which files changed, which
  functions were added, and how to run it.
- Run the code to check it works before saying it's done. If yfinance
  rate-limits, say so instead of guessing results.
- Sanity-check results (for example, an annual return can't be bigger than
  the total return). Point out anything that looks suspicious.
- Remind me to commit to Git after each finished step, and suggest a short
  commit message.
- Never invent data, results or event dates. Every number must come from
  running the code, and every event date needs an official source.

## Project structure

- `data.py`: reusable data and metric functions
  - `get_prices(ticker, start, end)`: adjusted daily closes via yfinance
  - `get_returns(prices)`: daily percent returns
  - `get_growth(prices, amount)`: final value of an investment
  - `get_equity(returns, amount=10000)`: daily portfolio value
  - `get_max_drawdown(equity)`: worst peak-to-trough drop
  - `get_annual_return(equity)`: compound annual growth rate (252 days/year)
  - `get_volatility(returns)`: annualized standard deviation
  - `get_sharpe(returns)`: annualized Sharpe ratio (risk-free rate = 0)
- `analyzer.py`: asks for a ticker, prints metrics, draws the equity chart
- `backtester.py`: 50/200-day moving-average crossover vs buy-and-hold
  - `signal.shift(1)` prevents lookahead bias (trade the day after a signal)
  - 0.1% transaction cost per trade
  - interactive code sits under `if __name__ == "__main__":`

## Conventions

- Functions live in modules; scripts import them.
- Chart style: dark background `#262624`, text `#C2C0B6`, pastel lines
  (`#A8C5DA` blue, `#E8B4A0` peach, `#B5D3B0` sage, `#D4B8E0` lavender).
- Labels must be accurate (portfolio value is not price).

## Results so far (QQQ, 2015-2025, before the fair-start fix)

| | Strategy | Buy & Hold |
|---|---|---|
| Annual return | 16.2% | 18.3% |
| Volatility | 18.0% | 21.8% |
| Sharpe | 0.92 | 0.88 |
| Max drawdown | -28.6% | -35.1% |
| Trades | 9 | - |

## Roadmap

1. [ ] Fair start date: both strategies begin when the 200-day MA exists
2. [ ] `robustness.py`: test the strategy across QQQ, SPY, NVDA, JPM, XOM, AAPL
3. [ ] Chip sector dashboard: metrics for NVDA, AMD, INTC, AVGO, QCOM, MU,
       TSM, ASML, AMAT, LRCX, KLAC, with SMH as the sector benchmark
4. [ ] Event list: major U.S. export-control announcements on advanced chips,
       each with its exact date and official source (e.g. Commerce Department
       press releases), stored in a CSV
5. [ ] Event study: abnormal returns (stock return minus market return) and
       cumulative abnormal returns over a window like [-5, +10] trading days
6. [ ] China exposure vs reaction: China revenue share from annual reports,
       compared with each stock's abnormal return
7. [ ] Monte Carlo simulator, including a prop firm challenge simulator
8. [ ] Streamlit web app with Plotly interactive charts
9. [ ] Later: portfolio optimizer (out-of-sample), options pricer and
       implied volatility smile, AI strategy input

## Background

This project is inspired by Prof. Lina Han's research (UMass Isenberg) on
how U.S. export controls affected the stock prices of American technology
suppliers, and on geoeconomic risk in the U.S. stock market.