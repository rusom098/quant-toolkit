Build a Streamlit web dashboard in app.py using the existing functions in data.py, backtester.py, robustness.py and chips.py. Don't duplicate their logic.

Layout: dark theme matching CLAUDE.md colors, sidebar with ticker input and date range, and 4 tabs:
1. Stock Analyzer: metric cards (total return, annual return, volatility, Sharpe, max drawdown) + interactive Plotly equity curve and drawdown chart.
2. Backtester: MA crossover vs buy-and-hold with sliders for short and long MA windows and transaction cost; metric cards side by side, Plotly chart of both equity curves with buy/sell markers.
3. Robustness: table across QQQ, SPY, NVDA, JPM, XOM, AAPL with the summary line.
4. Chip Sector: dashboard table sorted by Sharpe + interactive Plotly risk vs return scatter with SMH highlighted.

Cache downloads with st.cache_data so it's fast and doesn't get rate-limited. Add plotly and streamlit to a requirements.txt. Install what's needed, run it with "streamlit run app.py", check every tab loads without errors, then tell me how to open it and how to deploy it free on Streamlit Community Cloud from my GitHub repo.