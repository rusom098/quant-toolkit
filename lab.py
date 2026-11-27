"""
Options-Implied Distribution Lab: the main page of the Streamlit app.
All math is in implied_pdf.py and all downloading in sources.py; this file
only lays out the page and draws the charts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import implied_pdf as pdf
import sources
from theme import current_colors, style_chart, faded, PALETTE

SOURCES = ["Crypto options (Deribit)", "US stocks & ETFs (yfinance)", "CSV upload"]


# ---------- Cached loading ----------
# st.cache_data remembers results, so repeated page updates do not download
# again. Errors are not cached, so a failed download is retried next time.

@st.cache_data(ttl="10m", show_spinner=False)
def load_deribit(currency):
    return sources.load_deribit(currency)


@st.cache_data(ttl="1h", show_spinner=False)
def load_yfinance(ticker):
    return sources.load_yfinance(ticker)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return sources.load_csv(file_bytes)


@st.cache_data(show_spinner=False)
def run_analysis(market, expirations, rate, method):
    return pdf.analyze_market(market, expirations, rate, method)


# ---------- Formatting helpers ----------

# Money with sensible decimals: $61,214 for big prices, $149.30 for small ones.
def money(value):
    if value is None or np.isnan(value):
        return "-"
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def percent(value, decimals=1):
    return "-" if value is None or np.isnan(value) else f"{value:.{decimals}%}"


# Label for an expiration in drop-downs, e.g. "2026-12-18 (21.8 days)".
def expiration_label(result):
    return f"{result['expiration']} ({result['days']:.1f} days)"


# ---------- Section 1: inputs ----------

# Sidebar source choice plus the ticker / currency / upload. Returns a market or None.
def choose_market():
    st.sidebar.markdown("### Data source")
    source = st.sidebar.radio("Source", SOURCES, index=0, label_visibility="collapsed")
    if source == SOURCES[0]:
        currency = st.sidebar.selectbox("Currency", ["BTC", "ETH"])
        with st.spinner(f"Downloading {currency} options from Deribit..."):
            return load_deribit(currency)
    if source == SOURCES[1]:
        ticker = st.sidebar.text_input("Ticker", "NVDA").strip().upper()
        with st.spinner(f"Downloading {ticker} option chains from Yahoo Finance..."):
            return load_yfinance(ticker)

    st.sidebar.caption("Columns: strike, expiration (YYYY-MM-DD), type (call/put), bid, ask. "
                       "Optional: underlying_price, quote_date (YYYY-MM-DD).")
    with open(sources.SAMPLE_CSV, "rb") as file:
        sample_bytes = file.read()
    st.sidebar.download_button("Download sample CSV (SPY)", sample_bytes,
                               file_name="sample_options_SPY_2026-11-25.csv", mime="text/csv")
    upload = st.sidebar.file_uploader("Upload option quotes", type="csv")
    use_sample = st.sidebar.checkbox("Use the sample file instead", value=upload is None)
    if upload is None and not use_sample:
        st.info("Upload a CSV in the sidebar, or tick 'Use the sample file instead'.")
        return None
    file_bytes = sample_bytes if use_sample else upload.getvalue()
    return load_csv(file_bytes)


# Expirations, risk-free rate and smoothing method, in one row on the page.
def choose_settings(market):
    available = pdf.list_expirations(market)
    labels = {row.expiration: f"{row.expiration} ({row.days:.0f} d)" for row in available.itertuples()}
    chosen = st.pills("Expirations (7-90 days)", list(labels), selection_mode="multi",
                        default=list(labels), format_func=lambda e: labels[e])
    middle, right, _ = st.columns([1, 1, 2])
    rate = middle.number_input("Risk-free rate (%)", 0.0, 15.0, pdf.DEFAULT_RATE * 100, 0.25) / 100
    method = right.radio("Smoothing", pdf.SMOOTHING_METHODS, horizontal=True)
    return chosen, rate, method


# ---------- Section 2: data quality ----------

def show_data_quality(analysis):
    st.markdown("### 2. Data quality")
    table = pdf.quality_table(analysis)
    total = table.iloc[-1]
    dropped = int(total["Loaded"] - total["Kept"])
    columns = st.columns(4)
    columns[0].metric("Quotes loaded", f"{int(total['Loaded']):,}")
    columns[1].metric("Dropped", f"{dropped:,}")
    columns[2].metric("Kept", f"{int(total['Kept']):,}")
    columns[3].metric("Expirations passing checks",
                      f"{len(analysis['surface_days'])} of {len(analysis['results'])}")

    # Show only reasons that actually dropped something, to keep the table readable
    used_reasons = [reason for reason in pdf.DROP_REASONS if total[reason] > 0]
    shown = table[["Expiration", "Loaded"] + used_reasons + ["Kept", "IV points used"]]
    st.table(shown.set_index("Expiration").astype(int))
    st.caption("Kept = quotes with a valid implied vol. IV points used = one point per strike "
               "after choosing OTM puts/calls and blending them within ±5% of the forward.")

    for r in analysis["results"]:
        if r["warnings"]:
            st.warning(f"{r['expiration']} failed a sanity check and is left out of the surface: "
                       + "; ".join(r["warnings"]))


# ---------- Sections 3 and 4: smile and density charts ----------

def plot_smile(result, method):
    quotes = result["quotes"]
    good = quotes[quotes["reason"] == ""]
    fig = go.Figure()
    for kind, color, symbol in [("call", PALETTE[0], "circle"), ("put", PALETTE[1], "diamond")]:
        part = good[good["type"] == kind]
        fig.add_trace(go.Scatter(
            x=part["strike"], y=part["iv"], mode="markers", name=f"{kind}s (raw IV)",
            marker=dict(color=color, size=6, symbol=symbol, line=dict(width=0.5, color=current_colors()["text"])),
            hovertemplate="Strike %{x:,.2f}<br>IV %{y:.1%}<extra></extra>"))
    if "iv_grid" in result:
        fig.add_trace(go.Scatter(x=result["strike_grid"], y=result["iv_grid"], mode="lines",
                                 name=f"{method} fit", line=dict(color=PALETTE[2], width=2),
                                 hovertemplate="Strike %{x:,.2f}<br>Fit %{y:.1%}<extra></extra>"))
    fig.add_vline(x=result["forward"], line=dict(color=current_colors()["text"], width=1, dash="dot"),
                  annotation_text="forward", annotation_position="bottom right")
    fig.update_layout(title=f"Volatility smile, {expiration_label(result)}")
    fig.update_xaxes(title_text="Strike ($)")
    fig.update_yaxes(title_text="Implied volatility", tickformat=".0%")
    return style_chart(fig)


def plot_density(result, spot):
    colors = current_colors()
    density_per_percent = result["density"] * spot / 100  # per $1 -> per 1% of spot
    fig = go.Figure(go.Scatter(
        x=result["strikes"], y=density_per_percent, mode="lines", name="density",
        line=dict(color=PALETTE[0], width=2), fill="tozeroy", fillcolor=faded(PALETTE[0], 0.15),
        hovertemplate="$%{x:,.2f}<br>%{y:.4f} per 1% of spot<extra></extra>"))
    markers = [("p5", "5th", PALETTE[1]), ("p50", "50th", PALETTE[2]), ("p95", "95th", PALETTE[4])]
    for key, label, color in markers:
        fig.add_vline(x=result[key], line=dict(color=color, width=1.5, dash="dash"),
                      annotation_text=f"{label} {money(result[key])}", annotation_position="top",
                      annotation_font_color=colors["text"])
    fig.add_vline(x=spot, line=dict(color=colors["text"], width=1, dash="dot"),
                  annotation_text="spot", annotation_position="bottom right")
    fig.update_layout(title=f"Implied density, {expiration_label(result)}", showlegend=False)
    fig.update_xaxes(title_text="Price at expiration ($)")
    fig.update_yaxes(title_text="Probability per 1% of spot")
    return style_chart(fig)


# ---------- Section 5: 3D surface ----------

def plot_surface(analysis, name):
    colors = current_colors()
    scale = [[0, colors["bg"]], [0.25, PALETTE[0]], [0.6, PALETTE[2]], [1, PALETTE[3]]]
    fig = go.Figure(go.Surface(
        x=pdf.SURFACE_X, y=analysis["surface_days"], z=analysis["surface_z"],
        colorscale=scale, showscale=False, opacity=0.95,
        contours=dict(z=dict(show=False)),
        hovertemplate="%{x:.0f}% of spot<br>%{y:.1f} days<br>%{z:.4f} per 1%<extra></extra>"))
    axis = dict(backgroundcolor=colors["bg"], gridcolor=colors["grid"], linecolor=colors["border"],
                tickfont=dict(color=colors["text"]), title_font=dict(color=colors["text"]),
                showbackground=True, zerolinecolor=colors["border"])
    fig.update_layout(
        title=f"{name}: risk-neutral density across expirations",
        scene=dict(xaxis=dict(title="Strike (% of spot)", **axis),
                   yaxis=dict(title="Days to expiration", **axis),
                   zaxis=dict(title="Prob. per 1%", **axis),
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
    fig = style_chart(fig, height=600)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


# ---------- Section 6: probability calculator ----------

def show_calculator(results, spot):
    st.markdown("### 6. Probability calculator")
    labels = [expiration_label(r) for r in results]
    left, right = st.columns(2)
    chosen = left.selectbox("Expiration", labels, key="calc_expiration")
    result = results[labels.index(chosen)]
    price = right.number_input("Price X ($)", min_value=0.0, value=float(round(spot, 2)),
                               step=float(max(round(spot * 0.01, 2), 0.01)))
    above = pdf.prob_above(result, price)
    columns = st.columns(2)
    columns[0].metric(f"P(price above {money(price)})", percent(above))
    columns[1].metric(f"P(price below {money(price)})", percent(1 - above))
    low, high = result["strikes"][0], result["strikes"][-1]
    if not low <= price <= high:
        st.caption(f"{money(price)} is outside the strike range with data "
                   f"({money(low)}-{money(high)}), so this is a rough edge value.")


# ---------- Section 7: summary statistics ----------

def show_summary(results, spot):
    st.markdown("### 7. Summary statistics")
    rows = []
    for r in results:
        rows.append({
            "Expiration": expiration_label(r),
            "Forward": money(r["forward"]),
            "Mean": money(r["mean"]),
            "Mean vs fwd": f"{r['mean_gap']:+.2%}",
            "Std dev": money(r["std"]),
            "Skew": f"{r['skew']:+.2f}",
            # No "$" here: two "$" signs in one cell turn into a math formula in Streamlit
            "5th-95th pct ($)": f"{money(r['p5'])[1:]} to {money(r['p95'])[1:]}",
            "Median": money(r["p50"]),
            "P(> spot)": percent(r["p_above_spot"]),
            f"P(< -{pdf.TAIL_MOVE:.0%})": percent(r["p_down_tail"]),
            f"P(> +{pdf.TAIL_MOVE:.0%})": percent(r["p_up_tail"]),
            "Checks": "OK" if not r["warnings"] else "failed",
        })
    st.table(pd.DataFrame(rows).set_index("Expiration"))
    st.caption(f"Tails are measured from spot {money(spot)}. Mean vs fwd should be near 0%: "
               "under risk-neutral pricing the expected price equals the forward.")


# ---------- Section 8 and 9: caveats and download ----------

def show_caveats():
    st.markdown("### 8. How to read this")
    st.info(
        "**Risk-neutral, not real-world.** These probabilities are backed out of option prices. "
        "Those prices include a *fear premium*: traders pay extra for protection against crashes, "
        "so downside probabilities here are usually higher than what will really happen. "
        "This is a **distribution implied by today's prices, not a price prediction**. "
        "It also depends on the assumed risk-free rate, on no dividends, and on how the smile is "
        "smoothed, and it says nothing about prices outside the quoted strike range.")


def show_downloads(analysis, name):
    st.markdown("### 9. Download results")
    file_name = name.split(" ")[0].lower()
    left, right = st.columns(2)
    left.download_button("Summary by expiration (CSV)",
                         pdf.summary_table(analysis).to_csv(index=False),
                         file_name=f"{file_name}_implied_summary.csv", mime="text/csv")
    right.download_button("Density curves (CSV)",
                          pdf.density_table(analysis).to_csv(index=False),
                          file_name=f"{file_name}_implied_density.csv", mime="text/csv")


# ---------- Page ----------

def main():
    st.title("Options-Implied Distribution Lab")
    st.caption("What option prices say about the range of possible prices on each expiration "
               "date (Breeden-Litzenberger).")

    st.markdown("### 1. Inputs")
    try:
        market = choose_market()
    except Exception as error:
        st.error(f"Could not load data: {error} If this is a download, the service may be "
                 "rate-limiting; try again in a minute.")
        return
    if market is None:
        return
    chosen, rate, method = choose_settings(market)
    spot_text = money(market["spot"]) if market["spot"] is not None else "inferred"
    st.caption(f"{market['name']} · spot {spot_text} ({market['spot_note']}).")
    if not chosen:
        st.warning("Pick at least one expiration.")
        return

    analysis = run_analysis(market, tuple(chosen), rate, method)
    spot = analysis["spot"]
    show_data_quality(analysis)

    with_density = [r for r in analysis["results"] if "density" in r]
    if not with_density:
        st.error("No expiration had enough clean quotes to build a density.")
        return
    labels = [expiration_label(r) for r in with_density]
    default = int(np.argmin([abs(r["days"] - 30) for r in with_density]))  # closest to 30 days
    st.markdown("### 3. Volatility smile")
    chosen_label = st.selectbox("Expiration for the smile and density charts", labels, index=default)
    result = with_density[labels.index(chosen_label)]
    st.plotly_chart(plot_smile(result, method), width="stretch", theme=None)
    st.caption("Dots: our implied vols from bid-ask midpoints for every quote that passed the "
               "filters (in-the-money ones are shown but not used). Line: the smoothed fit.")

    st.markdown("### 4. Implied density")
    st.plotly_chart(plot_density(result, spot), width="stretch", theme=None)

    st.markdown("### 5. Density surface")
    if len(analysis["surface_days"]) >= 2:
        st.plotly_chart(plot_surface(analysis, market["name"]), width="stretch", theme=None)
        st.caption("Only expirations that passed every sanity check. Drag to rotate.")
    else:
        st.info("Fewer than 2 expirations passed the sanity checks, so there is no surface.")

    show_calculator(with_density, spot)
    show_summary(with_density, spot)
    show_caveats()
    show_downloads(analysis, market["name"])


main()
