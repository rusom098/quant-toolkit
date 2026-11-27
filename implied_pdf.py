"""
Options-implied probability distribution (Breeden-Litzenberger).

Idea in one sentence: option prices tell us how much the market charges for
each possible future price, and the curvature of call prices across strikes
turns that into a (risk-neutral) probability distribution.

Steps for each expiration:
1. Download out-of-the-money options (puts below spot, calls above spot).
2. Smooth their implied volatilities with a spline.
3. Turn the smooth IVs back into Black-Scholes call prices on a dense strike grid.
4. density = exp(rT) * (second derivative of call price with respect to strike).
"""
import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.stats import norm

from data import get_prices

# ---------- Settings ----------
RISK_FREE_RATE = 0.04      # assumed constant risk-free rate (4% per year), no dividends
MIN_DAYS = 7               # only use expirations at least this many days away...
MAX_DAYS = 90              # ...and at most this many days away
MIN_IV = 0.05              # drop implied volatilities below 5%...
MAX_IV = 3.00              # ...or above 300% (almost certainly bad data)
MIN_OPEN_INTEREST = 10     # drop options that almost nobody holds
STRIKE_BAND = (0.4, 2.0)   # only strikes from 40% to 200% of spot (far wings are worth pennies, so their IVs are noise)
MIN_OPTIONS = 10           # need at least this many good options to fit a curve
GRID_POINTS = 500          # number of strikes in the dense grid
IV_NOISE = 0.03           # assumed noise in Yahoo's IVs (3 vol points); sets how much the spline smooths

# Sanity-check tolerances
MASS_RANGE = (0.90, 1.10)  # raw density area (before renormalizing) should be near 1
MAX_NEGATIVE_SHARE = 0.05  # negative area may be at most 5% of the positive area
MAX_MEAN_GAP = 0.02        # mean must be within 2% of the forward price
PEAK_PROMINENCE = 0.05     # a "peak" must stick out by 5% of the tallest peak

# Surface x-axis: strike as % of spot
SURFACE_X = np.linspace(50, 150, 201)


# ---------- Downloading ----------

# Latest closing price, reusing get_prices from data.py.
def get_spot(ticker):
    today = datetime.date.today()
    start = str(today - datetime.timedelta(days=10))  # 10 days back covers weekends/holidays
    end = str(today + datetime.timedelta(days=1))     # end date is exclusive in yfinance
    prices = get_prices(ticker, start, end).dropna()
    return float(prices.iloc[-1]), prices.index[-1].date()


# Expiration dates between MIN_DAYS and MAX_DAYS from today, as (date text, days) pairs.
def get_expirations(ticker):
    today = datetime.date.today()
    chosen = []
    for text in yf.Ticker(ticker).options:
        expiry = datetime.date.fromisoformat(text)
        days = (expiry - today).days
        if MIN_DAYS <= days <= MAX_DAYS:
            chosen.append((text, days))
    return chosen


# Out-of-the-money options for one expiration, with bad rows removed.
# Returns a table with columns strike, iv, open_interest, type (sorted by strike).
def get_otm_options(ticker, expiration, spot):
    chain = yf.Ticker(ticker).option_chain(expiration)
    puts = chain.puts[chain.puts["strike"] < spot].copy()
    calls = chain.calls[chain.calls["strike"] >= spot].copy()
    puts["type"] = "put"
    calls["type"] = "call"
    options = pd.concat([puts, calls])

    options = options.rename(columns={"impliedVolatility": "iv", "openInterest": "open_interest"})
    options = options[["strike", "iv", "open_interest", "type"]]
    options = options.dropna(subset=["strike", "iv"])
    options["open_interest"] = options["open_interest"].fillna(0)

    good_iv = (options["iv"] >= MIN_IV) & (options["iv"] <= MAX_IV)
    enough_interest = options["open_interest"] >= MIN_OPEN_INTEREST
    low, high = STRIKE_BAND
    near_spot = (options["strike"] >= low * spot) & (options["strike"] <= high * spot)
    options = options[good_iv & enough_interest & near_spot]
    return options.sort_values("strike").reset_index(drop=True)


# ---------- Math ----------

# Black-Scholes price of a European call (no dividends). Works on numpy arrays.
def bs_call_price(spot, strike, years, rate, vol):
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * years) / (vol * np.sqrt(years))
    d2 = d1 - vol * np.sqrt(years)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * years) * norm.cdf(d2)


# Fit a smooth IV curve through the options and evaluate it on a dense strike grid.
# s = n * noise^2 lets the curve miss each point by about IV_NOISE on average.
# Less smoothing follows the noise, and the second derivative then turns spiky.
def smooth_iv(strikes, ivs):
    smoothing = len(strikes) * IV_NOISE ** 2
    spline = UnivariateSpline(strikes, ivs, k=3, s=smoothing)  # k=3: cubic spline
    strike_grid = np.linspace(strikes.min(), strikes.max(), GRID_POINTS)
    iv_grid = spline(strike_grid)
    iv_grid = np.clip(iv_grid, MIN_IV, MAX_IV)  # keep the smoothed curve in a sensible range
    return strike_grid, iv_grid


# Breeden-Litzenberger: density = exp(rT) * d²C/dK², using central finite differences.
# The first and last grid points are lost (they have no neighbor on one side).
def raw_density(strike_grid, call_prices, years, rate):
    step = strike_grid[1] - strike_grid[0]
    second_diff = (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / step ** 2
    density = np.exp(rate * years) * second_diff
    return strike_grid[1:-1], density


# Price at a given cumulative probability (e.g. 0.05 for the 5th percentile).
def percentile_price(strikes, cdf, probability):
    return float(np.interp(probability, cdf, strikes))


# ---------- One expiration ----------

# Build the density for one expiration and run the sanity checks.
# Returns a dictionary with the results and a list of warnings (empty = all good).
def analyze_expiration(ticker, expiration, days, spot):
    years = days / 365  # calendar days, the usual convention for options
    forward = spot * np.exp(RISK_FREE_RATE * years)
    result = {"expiration": expiration, "days": days, "forward": forward, "warnings": []}

    options = get_otm_options(ticker, expiration, spot)
    result["n_options"] = len(options)
    if len(options) < MIN_OPTIONS:
        result["warnings"].append(f"only {len(options)} valid options (need {MIN_OPTIONS})")
        return result

    strikes = options["strike"].to_numpy(dtype=float)
    ivs = options["iv"].to_numpy(dtype=float)
    strike_grid, iv_grid = smooth_iv(strikes, ivs)
    call_prices = bs_call_price(spot, strike_grid, years, RISK_FREE_RATE, iv_grid)
    density_strikes, density = raw_density(strike_grid, call_prices, years, RISK_FREE_RATE)

    # Sanity numbers measured BEFORE cleaning
    positive_area = trapezoid(np.clip(density, 0, None), density_strikes)
    negative_area = -trapezoid(np.clip(density, None, 0), density_strikes)
    raw_mass = trapezoid(density, density_strikes)  # share of probability inside the strike range

    # Clean: negative probabilities are impossible, so set them to 0, then rescale to area 1
    density = np.clip(density, 0, None)
    density = density / trapezoid(density, density_strikes)

    mean = trapezoid(density_strikes * density, density_strikes)
    cdf = cumulative_trapezoid(density, density_strikes, initial=0)
    peaks, _ = find_peaks(density, prominence=PEAK_PROMINENCE * density.max())

    result.update({
        "options": options,
        "strike_grid": strike_grid, "iv_grid": iv_grid,
        "strikes": density_strikes, "density": density,
        "raw_mass": raw_mass,
        "negative_share": negative_area / positive_area,
        "mean": mean,
        "mean_gap": mean / forward - 1,
        "n_peaks": len(peaks),
        "p5": percentile_price(density_strikes, cdf, 0.05),
        "p95": percentile_price(density_strikes, cdf, 0.95),
        "prob_above_spot": 1 - float(np.interp(spot, density_strikes, cdf)),
    })
    result["warnings"] = check_density(result)
    return result


# Compare one expiration's numbers with the tolerances. Returns a list of warnings.
def check_density(result):
    warnings = []
    low, high = MASS_RANGE
    if not low <= result["raw_mass"] <= high:
        warnings.append(f"raw density area is {result['raw_mass']:.2f} (expected {low}-{high}); "
                        "the strike range may miss part of the distribution")
    if result["negative_share"] > MAX_NEGATIVE_SHARE:
        warnings.append(f"negative region is {result['negative_share']:.1%} of the positive area")
    if abs(result["mean_gap"]) > MAX_MEAN_GAP:
        warnings.append(f"mean is {result['mean_gap']:+.1%} away from the forward price")
    if result["n_peaks"] > 1:
        warnings.append(f"density has {result['n_peaks']} peaks (spiky)")
    return warnings


# ---------- Whole ticker ----------

# Density on the shared surface grid (strike as % of spot), in probability per 1% of spot.
# Outside the strike range of the data the value is 0.
def density_on_surface_grid(result, spot):
    percent_of_spot = result["strikes"] / spot * 100
    density_per_percent = result["density"] * spot / 100  # per $1 -> per 1% of spot
    return np.interp(SURFACE_X, percent_of_spot, density_per_percent, left=0, right=0)


# Run every expiration for one ticker. Returns a dictionary with spot, the
# per-expiration results, and the surface built from expirations that passed.
def analyze_ticker(ticker):
    spot, spot_date = get_spot(ticker)
    results = []
    for expiration, days in get_expirations(ticker):
        results.append(analyze_expiration(ticker, expiration, days, spot))

    passed = [r for r in results if not r["warnings"]]
    surface_rows = [density_on_surface_grid(r, spot) for r in passed]
    return {
        "ticker": ticker, "spot": spot, "spot_date": spot_date,
        "results": results,
        "surface_days": [r["days"] for r in passed],
        "surface_z": np.array(surface_rows),
    }


# One table row per expiration: price range, probability above spot, checks.
def summary_table(analysis):
    rows = []
    for r in analysis["results"]:
        row = {"Expiration": r["expiration"], "Days": r["days"], "Options used": r["n_options"]}
        if "density" in r:
            row["5th pct"] = r["p5"]
            row["95th pct"] = r["p95"]
            row["P(above spot)"] = r["prob_above_spot"]
            row["Mean vs forward"] = r["mean_gap"]
            row["Raw area"] = r["raw_mass"]
        row["Checks"] = "OK" if not r["warnings"] else "; ".join(r["warnings"])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ticker = input("Ticker (default NVDA): ").strip().upper() or "NVDA"
    analysis = analyze_ticker(ticker)
    print(f"\n{ticker} spot: ${analysis['spot']:.2f} (close of {analysis['spot_date']})")
    print(f"Risk-free rate assumed: {RISK_FREE_RATE:.0%}, no dividends\n")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(summary_table(analysis).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n{len(analysis['surface_days'])} of {len(analysis['results'])} expirations passed "
          "all checks and are used in the surface.")
