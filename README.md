# Options-Implied Distribution Lab

> 🚧 **Work in progress.** The core pipeline works; the interface and research write-up are being built.

A research tool that reads **option prices** and turns them into the **market's implied probability distribution** of an asset's future price, using the **Breeden-Litzenberger** method.

Instead of a single price prediction, it answers questions like:

- *What's the market-implied chance BTC is above $70,000 on a given date?*
- *What price range does the market expect over the next month?*
- *How much crash risk is priced in compared with upside?*

<!-- Add once deployed: **Live demo:** https://your-app.streamlit.app -->
<!-- Add a screenshot: ![Lab screenshot](docs/screenshot.png) -->

---

## How it works

Taking a second derivative amplifies noise, so most of the pipeline exists to clean and smooth the data first.

1. **Load option quotes:** bid and ask prices across strikes and expirations.
2. **Clean the data:** drop zero bids, very wide spreads, and quotes that violate no-arbitrage rules (call prices must fall and be convex as the strike rises).
3. **Compute implied volatility** for each option from its bid-ask midpoint (Black-76, i.e. Black-Scholes on the forward price, solved numerically with `brentq`).
4. **Smooth the volatility smile** with a spline or an SVI fit, using out-of-the-money options and blending puts and calls near the money.
5. **Convert back to call prices** on a fine strike grid.
6. **Breeden-Litzenberger:** the risk-neutral density is

   $$f(K) = e^{rT}\,\frac{\partial^2 C(K,T)}{\partial K^2}$$

7. **Sanity-check every distribution:** it must integrate to about 1, its mean must sit close to the forward price, and it must have no spikes. Expirations that fail are flagged and excluded.

## Data sources

| Source | Covers | Notes |
|---|---|---|
| Deribit public API | BTC, ETH options | Live, European-style options (default) |
| yfinance | US stocks and ETFs | Delayed; US single-stock options are American-style, so results are an approximation |
| CSV upload | Anything | Columns: `strike, expiration, type, bid, ask`; optional `underlying_price`, `quote_date`. Without a price, the forward comes from put-call parity |

## Example snapshot

BTC (Deribit), captured 27 Nov 2026, 28-day expiration. Results change with every market session.

| Metric | Value |
|---|---|
| Quotes loaded / kept | 378 / 329 |
| Expirations passing sanity checks | 4 of 5 |
| 90% implied range (5th–95th percentile) | $47,850 – $75,191 |
| Mean vs. forward price | +0.18% |
| P(price falls more than 20%) | 6.4% |
| P(price rises more than 20%) | 7.0% |

## Status

**Done**
- [x] Data loading from Deribit, yfinance and CSV
- [x] Own implied-volatility calculation, arbitrage filtering, put/call blending
- [x] Spline and SVI smoothing
- [x] Density, percentiles, probability calculator, 3D surface across expirations
- [x] Sanity checks with automatic exclusion of failing expirations
- [x] CSV export of results

**In progress**
- [ ] Answer-first page layout (ask a question, see the probability)
- [ ] Interface fixes (3D chart scrolling, icons)
- [ ] Investigate tail coverage at longer expirations
- [ ] Research notebook with findings and methodology
- [ ] Deployment on Streamlit Community Cloud

**Planned**
- [ ] Compare distributions before and after major events (e.g. U.S. export-control announcements for semiconductor stocks)
- [ ] "Your view vs. the market" comparison

## Other tools in this repo

The **Other tools** page contains earlier work from this project:

- **Stock analyzer:** returns, volatility, Sharpe ratio and max drawdown for any ticker.
- **Moving-average backtester and robustness test:** a 50/200-day crossover with transaction costs, lookahead-bias controls and aligned start dates. Across 6 tickers (2015–2025) it beat buy-and-hold's Sharpe ratio on only 2, so its edge did not generalize.
- **Semiconductor sector dashboard:** 11 chip stocks compared with the SMH ETF on risk-adjusted performance.

## Run it locally

```bash
git clone https://github.com/rusom098/quant-toolkit.git
cd quant-toolkit
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

To run the pipeline without the app and print the data-quality and sanity-check tables:

```bash
python implied_pdf.py
```

## Project structure

```
app.py            # Entry point: theme switcher and page navigation
lab.py            # Main page: the Options-Implied Distribution Lab
other_tools.py    # Second page: analyzer, backtester, robustness, chip sector tabs
implied_pdf.py    # Breeden-Litzenberger pipeline and sanity checks (no Streamlit)
sources.py        # Data loaders: Deribit, yfinance, CSV (with validation)
theme.py          # Light / Mid grey / Dark themes and chart styling
data.py           # Price data and performance metrics
analyzer.py       # Command-line stock analyzer
backtester.py     # Moving-average crossover backtester
robustness.py     # Multi-ticker robustness test
chips.py          # Semiconductor sector dashboard
samples/          # Example option-chain CSV (real SPY snapshot, 25 Nov 2026)
```

## Limitations

- **Risk-neutral, not real-world.** Option prices include a premium for crash protection, so these probabilities overstate downside compared with what traders actually expect. This is a distribution implied by today's prices, not a forecast.
- Results depend on the assumed risk-free rate, ignore dividends, and say nothing about prices outside the quoted strike range.
- Free data can be delayed or stale, especially for stocks via yfinance.

## Disclaimer

For research and educational purposes only. Not financial advice.

---

*Built by Ruso Meskhishvili, Computer Engineering, University of Massachusetts Amherst.*