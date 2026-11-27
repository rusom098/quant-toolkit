Build a Streamlit web dashboard in app.py using the existing functions in data.py, backtester.py, robustness.py and chips.py. Don't duplicate their logic.

Layout: dark theme matching CLAUDE.md colors, sidebar with ticker input and date range, and 4 tabs:
1. Stock Analyzer: metric cards (total return, annual return, volatility, Sharpe, max drawdown) + interactive Plotly equity curve and drawdown chart.
2. Backtester: MA crossover vs buy-and-hold with sliders for short and long MA windows and transaction cost; metric cards side by side, Plotly chart of both equity curves with buy/sell markers.
3. Robustness: table across QQQ, SPY, NVDA, JPM, XOM, AAPL with the summary line.
4. Chip Sector: dashboard table sorted by Sharpe + interactive Plotly risk vs return scatter with SMH highlighted.

Cache downloads with st.cache_data so it's fast and doesn't get rate-limited. Add plotly and streamlit to a requirements.txt. Install what's needed, run it with "streamlit run app.py", check every tab loads without errors, then tell me how to open it and how to deploy it free on Streamlit Community Cloud from my GitHub repo.

Build a new feature: options-implied probability distribution (Breeden-Litzenberger), as a new dashboard tab "Implied Distribution". Put the logic in a new file implied_pdf.py and keep app.py changes small.

Method:
1. Use yfinance option chains for a selected ticker (default NVDA). Get the spot price and expirations between 7 and 90 days out.
2. For each expiration, use out-of-the-money options only: puts below spot, calls above spot. Drop options with zero/missing implied volatility, implausible IV (below 5% or above 300%), or very low open interest.
3. Fit a smoothing spline of implied volatility vs strike (scipy), and evaluate it on a dense strike grid within the range of the data.
4. Convert the smoothed IVs back to Black-Scholes call prices (risk-free rate as a clearly labeled constant, e.g. 4%; no dividends).
5. Risk-neutral density = exp(rT) * second derivative of call price with respect to strike (finite differences). Set small negative values to 0 and renormalize so each density integrates to 1.

Sanity checks (print them and show warnings in the app):
- Each density integrates to ~1, has no large negative regions, and its mean is close to the forward price S*exp(rT).
- Flag expirations whose density has spikes or too few valid options, and exclude them from the surface.

Style:
- Theme switcher in the sidebar with three backgrounds:
  1) Light: #E7E7E7 (DEFAULT)
  2) Mid grey: #868686
  3) Dark: #0D0D0D
- Text color adapts: near-black text (#111111) on Light and Mid grey, near-white text (#EDEDED) on Dark. Gridlines and borders are subtle versions of the text color.
- Charts are colorful: use a vivid, colorblind-friendly palette in this order:
  #2563EB (blue), #E4572E (orange-red), #17A398 (teal), #F2A541 (amber), #7B2CBF (purple), #D7263D (red).
  Chart backgrounds match the page background.
- Otherwise keep the clean research-paper look: thin lines, no glows or decorative elements.
- Check every chart is readable on all three backgrounds and adjust line widths or colors if anything is hard to see.

Add scipy to requirements.txt if needed, cache downloads, run it, check the sanity checks pass for NVDA, update CLAUDE.md, and summarize results and caveats. Note: markets may be closed, so option data could be from the last trading session.