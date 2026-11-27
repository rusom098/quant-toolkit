import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import get_prices, get_returns, get_equity, get_drawdown, get_summary
from backtester import compare_strategies, get_positions
from robustness import TICKERS, test_ticker, count_wins
from chips import CHIP_TICKERS, BENCHMARK, get_stock_metrics, sort_by_sharpe, get_verdict
from theme import current_colors, PALETTE

# Colors follow the theme chosen in the sidebar (see theme.py)
COLORS = current_colors()
BG = COLORS["bg"]
TEXT = COLORS["text"]
GRID = COLORS["grid"]
BLUE, PEACH, SAGE, LAVENDER = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]

CACHE_TIME = "12h"  # keep downloads for 12 hours so we don't hit yfinance limits
RATE_LIMIT_HINT = "This can happen if the ticker is wrong or yfinance is rate-limiting. Try again in a minute."


# ---------- Cached data loading ----------
# st.cache_data remembers each result, so the same download only happens once.
# If a function raises an error, Streamlit does NOT cache it, so a failed
# download (e.g. from a rate limit) will be retried next time.

# Download prices for one ticker (cached). Raises an error if nothing came back.
@st.cache_data(ttl=CACHE_TIME, show_spinner=False)
def load_prices(ticker, start, end):
    prices = get_prices(ticker, start, end).dropna()
    if prices.empty:
        raise ValueError(f"No price data for {ticker}.")
    return prices


# Run the robustness test for one ticker (cached).
@st.cache_data(ttl=CACHE_TIME, show_spinner=False)
def load_robustness_row(ticker, start, end):
    row = test_ticker(ticker, start, end)
    if row is None:
        raise ValueError(f"No price data for {ticker}.")
    return row


# Get the chip dashboard metrics for one ticker (cached).
@st.cache_data(ttl=CACHE_TIME, show_spinner=False)
def load_chip_metrics(ticker, start, end):
    row = get_stock_metrics(ticker, start, end)
    if row is None:
        raise ValueError(f"No price data for {ticker}.")
    return row


# ---------- Small display helpers ----------

# Apply the theme's colors to a Plotly chart.
def style_chart(fig, height=500):
    fig.update_layout(
        template=COLORS["template"],
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=10, r=10, t=100, b=10),  # room at the top for title + legend
        hovermode="x unified",
        title=dict(y=0.98, yanchor="top"),  # title at the very top...
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),  # ...legend just above the plot
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# Turn a get_summary() dictionary into (label, text) pairs for metric cards.
def summary_cards(summary):
    cards = [
        ("Total return", f"{summary['total']:.1%}"),
        ("Annual return", f"{summary['annual']:.1%}"),
        ("Volatility", f"{summary['volatility']:.1%}"),
        ("Sharpe ratio", f"{summary['sharpe']:.2f}"),
        ("Max drawdown", f"{summary['max_dd']:.1%}"),
    ]
    return cards


# Show metric cards in rows of `per_row` cards.
def show_cards(cards, per_row):
    for i in range(0, len(cards), per_row):
        columns = st.columns(per_row)
        for column, (label, value) in zip(columns, cards[i:i + per_row]):
            column.metric(label, value)


# ---------- Tab 1: Stock Analyzer ----------

# Equity curve on top, drawdown underneath, sharing the same dates.
def plot_equity_and_drawdown(equity, drawdown, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.08,
                        subplot_titles=(f"Growth of $10,000 in {ticker}", "Drawdown from previous peak"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Portfolio value",
                             line=dict(color=BLUE, width=2), hovertemplate="$%{y:,.0f}"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill="tozeroy",
                             line=dict(color=PEACH, width=1.5), hovertemplate="%{y:.1%}"),
                  row=2, col=1)
    fig.update_yaxes(title_text="Portfolio value ($)", tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig = style_chart(fig, height=650)
    fig.update_annotations(font_color=TEXT)  # subplot titles
    return fig


def show_analyzer_tab(ticker, start, end):
    prices = load_prices(ticker, start, end)
    returns = get_returns(prices)
    st.subheader(f"{ticker}: {prices.index[0].date()} to {prices.index[-1].date()}")
    show_cards(summary_cards(get_summary(returns)), per_row=5)

    equity = get_equity(returns)
    drawdown = get_drawdown(equity)
    st.plotly_chart(plot_equity_and_drawdown(equity, drawdown, ticker), width="stretch", theme=None)


# ---------- Tab 2: Backtester ----------

# Both equity curves, with triangles where the strategy buys and sells.
def plot_backtest(strat_equity, bh_equity, position, short, long):
    changes = position.diff()  # +1 = bought, -1 = sold
    buy_dates = changes[changes == 1].index
    sell_dates = changes[changes == -1].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity, name=f"MA crossover ({short}/{long})",
                             line=dict(color=BLUE, width=2), hovertemplate="$%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="Buy & hold",
                             line=dict(color=PEACH, width=2), hovertemplate="$%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=buy_dates, y=strat_equity[buy_dates], name="Buy", mode="markers",
                             marker=dict(symbol="triangle-up", size=12, color=SAGE, line=dict(color=BG, width=2)),
                             hovertemplate="Buy"))
    fig.add_trace(go.Scatter(x=sell_dates, y=strat_equity[sell_dates], name="Sell", mode="markers",
                             marker=dict(symbol="triangle-down", size=12, color=LAVENDER, line=dict(color=BG, width=2)),
                             hovertemplate="Sell"))
    fig.update_layout(title="Portfolio value: strategy vs buy & hold")
    fig.update_yaxes(title_text="Portfolio value ($)", tickprefix="$", tickformat=",.0f")
    return style_chart(fig)


def show_backtester_tab(ticker, start, end):
    col1, col2, col3 = st.columns(3)
    short = col1.slider("Short MA window (days)", 10, 100, 50, step=5)
    long = col2.slider("Long MA window (days)", 120, 250, 200, step=10)
    cost_percent = col3.slider("Transaction cost per trade (%)", 0.0, 1.0, 0.1, step=0.05)
    cost = cost_percent / 100  # 0.1% -> 0.001

    prices = load_prices(ticker, start, end)
    if len(prices) <= long + 20:  # need the long MA plus some days to trade
        st.warning(f"Not enough data: {len(prices)} trading days, but the long MA needs {long}. "
                   "Pick an earlier start date.")
        return

    strategy, buy_hold, trades = compare_strategies(prices, short, long, cost)
    position = get_positions(prices, short, long)
    st.caption(f"Both strategies start on {strategy.index[0].date()}, "
               f"the first day the {long}-day MA exists (fair start).")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**MA crossover ({short}/{long})**")
        show_cards(summary_cards(get_summary(strategy)) + [("Trades", str(trades))], per_row=3)
    with right:
        st.markdown("**Buy & hold**")
        show_cards(summary_cards(get_summary(buy_hold)) + [("Trades", "1")], per_row=3)

    strat_equity = get_equity(strategy)
    bh_equity = get_equity(buy_hold)
    st.plotly_chart(plot_backtest(strat_equity, bh_equity, position, short, long), width="stretch", theme=None)


# ---------- Tab 3: Robustness ----------

# One table row per ticker. Percentages are multiplied by 100 for display.
def robustness_table(rows):
    table = []
    for row in rows:
        s = row["strategy"]
        b = row["buy_hold"]
        table.append({
            "Ticker": row["ticker"],
            "Start": str(row["start"]),
            "Annual (strategy)": s["annual"] * 100,
            "Annual (B&H)": b["annual"] * 100,
            "Sharpe (strategy)": s["sharpe"],
            "Sharpe (B&H)": b["sharpe"],
            "Max DD (strategy)": s["max_dd"] * 100,
            "Max DD (B&H)": b["max_dd"] * 100,
            "Trades": row["trades"],
        })
    return pd.DataFrame(table)


def show_robustness_tab(start, end):
    st.subheader("Does the 50/200 MA crossover work beyond QQQ?")
    st.caption("Fixed 50/200-day windows and 0.1% cost (the sliders in the Backtester tab don't apply here).")

    rows = []
    with st.spinner("Running the backtest on each ticker..."):
        for ticker in TICKERS:
            try:
                rows.append(load_robustness_row(ticker, start, end))
            except Exception as error:
                st.error(f"{ticker}: {error} {RATE_LIMIT_HINT}")
    if not rows:
        return

    percent = st.column_config.NumberColumn(format="%.1f%%")
    two_decimals = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(robustness_table(rows), hide_index=True, width="stretch", column_config={
        "Annual (strategy)": percent, "Annual (B&H)": percent,
        "Max DD (strategy)": percent, "Max DD (B&H)": percent,
        "Sharpe (strategy)": two_decimals, "Sharpe (B&H)": two_decimals,
    })

    wins = count_wins(rows)
    total = len(rows)
    st.markdown(f"The strategy had a **higher Sharpe on {wins['sharpe']}/{total}** tickers, "
                f"a **smaller max drawdown on {wins['max_dd']}/{total}**, "
                f"and a **higher annual return on {wins['annual']}/{total}**.")


# ---------- Tab 4: Chip Sector ----------

# Dashboard table: stocks sorted by Sharpe, with SMH as the last row.
def chip_table(rows, benchmark_row):
    table = []
    for row in sort_by_sharpe(rows) + [benchmark_row]:
        table.append({
            "Ticker": row["ticker"],
            "Total return": row["total"] * 100,
            "Annual return": row["annual"] * 100,
            "Volatility": row["volatility"] * 100,
            "Sharpe": row["sharpe"],
            "Max drawdown": row["max_dd"] * 100,
            "vs SMH": get_verdict(row, benchmark_row),
        })
    return pd.DataFrame(table)


# Hover text for one dot in the scatter chart.
def scatter_hover():
    return ("<b>%{text}</b><br>Annual return: %{y:.1f}%<br>Volatility: %{x:.1f}%"
            "<br>Total return: %{customdata[0]:,.0f}%<br>Sharpe: %{customdata[1]:.2f}"
            "<br>Max drawdown: %{customdata[2]:.1f}%<extra></extra>")


# Risk (x) vs return (y) scatter, one dot per stock, SMH in peach.
def plot_risk_return(rows, benchmark_row):
    fig = go.Figure()
    groups = [(rows, "Chip stocks", BLUE, 11, "top right"),
              ([benchmark_row], "SMH (benchmark)", PEACH, 15, "bottom right")]
    for group_rows, name, color, size, label_position in groups:
        fig.add_trace(go.Scatter(
            x=[row["volatility"] * 100 for row in group_rows],
            y=[row["annual"] * 100 for row in group_rows],
            text=[row["ticker"] for row in group_rows],
            customdata=[[row["total"] * 100, row["sharpe"], row["max_dd"] * 100] for row in group_rows],
            name=name, mode="markers+text", textposition=label_position,
            textfont=dict(color=TEXT),
            marker=dict(size=size, color=color, line=dict(color=BG, width=2)),
            hovertemplate=scatter_hover(),
        ))
    # Dashed line at SMH's return: dots above it beat the sector
    fig.add_hline(y=benchmark_row["annual"] * 100, line_dash="dash", line_color=PEACH, opacity=0.5,
                  annotation_text="SMH annual return", annotation_font_color=TEXT)
    fig.update_layout(title="Annual return vs volatility", hovermode="closest")
    fig.update_xaxes(title_text="Annualized volatility (%)")
    fig.update_yaxes(title_text="Annual return, CAGR (%)")
    return style_chart(fig, height=600)


def show_chips_tab(start, end):
    st.subheader("Chip sector dashboard vs SMH")

    rows = []
    with st.spinner("Downloading chip stocks..."):
        for ticker in CHIP_TICKERS:
            try:
                rows.append(load_chip_metrics(ticker, start, end))
            except Exception as error:
                st.error(f"{ticker}: {error} {RATE_LIMIT_HINT}")
        try:
            benchmark_row = load_chip_metrics(BENCHMARK, start, end)
        except Exception as error:
            st.error(f"{BENCHMARK}: {error} {RATE_LIMIT_HINT} There is no benchmark to compare against.")
            return
    if not rows:
        return

    percent = st.column_config.NumberColumn(format="%.1f%%")
    table = chip_table(rows, benchmark_row)
    table_height = (len(table) + 1) * 35 + 3  # 35 px per row (+ header), so no scrolling is needed
    st.dataframe(table, hide_index=True, width="stretch", height=table_height, column_config={
        "Total return": st.column_config.NumberColumn(format="%.0f%%"),
        "Annual return": percent, "Volatility": percent, "Max drawdown": percent,
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption("Sorted by Sharpe ratio. 'beat' means a higher annual return than SMH. "
               "Note: these are today's big chip companies (survivorship bias), and SMH holds most of them.")
    st.plotly_chart(plot_risk_return(rows, benchmark_row), width="stretch", theme=None)


# ---------- Page layout ----------

# Sidebar inputs. Returns the ticker and the start/end dates as text.
def sidebar_inputs():
    st.sidebar.subheader("Other tools")
    ticker = st.sidebar.text_input("Ticker (Analyzer & Backtester tabs)", "QQQ").strip().upper()
    start = st.sidebar.date_input("Start date", datetime.date(2015, 1, 1))
    end = st.sidebar.date_input("End date", datetime.date(2025, 1, 1))
    st.sidebar.caption("Adjusted daily closes from Yahoo Finance via yfinance. "
                       "Downloads are cached for 12 hours.")
    return ticker, start, end


# Run one tab's code and show any error on the page instead of crashing.
def run_safely(tab_function, *args):
    try:
        tab_function(*args)
    except Exception as error:
        st.error(f"Something went wrong: {error} {RATE_LIMIT_HINT}")


def main():
    ticker, start, end = sidebar_inputs()

    # The 200-day MA needs ~10 months of data before the backtest can start
    if (end - start).days < 730:
        st.error("Please choose a date range of at least 2 years.")
        st.stop()
    start = str(start)  # yfinance wants dates as text like "2015-01-01"
    end = str(end)

    analyzer, backtester, robustness, chips = st.tabs(
        ["Stock Analyzer", "Backtester", "Robustness", "Chip Sector"])
    with analyzer:
        run_safely(show_analyzer_tab, ticker, start, end)
    with backtester:
        run_safely(show_backtester_tab, ticker, start, end)
    with robustness:
        run_safely(show_robustness_tab, start, end)
    with chips:
        run_safely(show_chips_tab, start, end)


main()
