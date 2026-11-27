"""
Options-implied probability distribution (Breeden-Litzenberger).

Idea in one sentence: option prices tell us how much the market charges for
each possible future price, and the curvature of call prices across strikes
turns that into a (risk-neutral) probability distribution.

This file only does math. It works on a standard "market" dictionary made by
sources.py, so every data source (yfinance, Deribit, CSV) goes through the
same pipeline:

1. Clean quotes: drop zero bids, very wide spreads and strikes far from spot.
2. Drop arbitrage violations (call prices must fall and be convex in strike).
3. Compute our own implied vols from bid-ask midpoints (Black-76, brentq).
4. Use OTM puts below the forward and OTM calls above it, blending the two
   within +-5% of the forward.
5. Smooth the IVs (spline or SVI) and turn them back into call prices.
6. density = second derivative of the (undiscounted) call price in strike.
   This is the same as exp(rT) * d²C/dK² for discounted prices.
7. Sanity checks and summary statistics.

All prices are in "undiscounted" form (today's price * exp(rT)), which lets
us use Black-76 with the forward price and skip discounting everywhere else.
"""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, least_squares
from scipy.signal import find_peaks
from scipy.stats import norm

# ---------- Settings ----------
DEFAULT_RATE = 0.04        # default risk-free rate (4% per year); the user can change it
MIN_DAYS = 7               # only use expirations at least this many days away...
MAX_DAYS = 90              # ...and at most this many days away
MAX_SPREAD = 0.50          # drop quotes whose (ask - bid) / mid is above 50%
STRIKE_BAND = (0.4, 2.0)   # only strikes from 40% to 200% of the forward
MIN_IV = 0.05              # implied vols below 5%...
MAX_IV = 3.00              # ...or above 300% are treated as bad data
BLEND_BAND = 0.05          # blend put and call IVs within +-5% of the forward
MIN_OPTIONS = 10           # need at least this many good IV points to fit a curve
GRID_POINTS = 500          # number of strikes in the dense grid
IV_NOISE = 0.03            # spline smoothing: let the curve miss each point by ~3 vol points
SMOOTHING_METHODS = ["Spline", "SVI"]

# Sanity-check tolerances
MASS_RANGE = (0.90, 1.10)  # raw density area (before renormalizing) should be near 1
MAX_NEGATIVE_SHARE = 0.05  # negative area may be at most 5% of the positive area
MAX_MEAN_GAP = 0.02        # mean must be within 2% of the forward price
PEAK_PROMINENCE = 0.05     # a "peak" must stick out by 5% of the tallest peak
TAIL_MOVE = 0.20           # tail probabilities: price below -20% or above +20% of spot

# Surface x-axis: strike as % of spot
SURFACE_X = np.linspace(50, 150, 201)

# Reasons a quote can be dropped, in the order the filters run
DROP_REASONS = ["missing or zero bid", "wide spread", "strike far from forward",
                "price not decreasing (arbitrage)", "price not convex (arbitrage)",
                "no IV solution", "IV outside 5%-300%"]


# ---------- Time ----------

# Years between two datetimes (365-day years, the usual convention for options).
def years_between(start, end):
    seconds = (end - start).total_seconds()
    return seconds / (365 * 24 * 3600)


# One row per expiration with days and years left, keeping only MIN_DAYS to MAX_DAYS.
def list_expirations(market):
    rows = []
    for expiration, group in market["quotes"].groupby("expiration"):
        expiry_time = group["expiry_time"].iloc[0]
        years = years_between(market["as_of"], expiry_time)
        days = years * 365
        if MIN_DAYS <= days <= MAX_DAYS:
            rows.append({"expiration": expiration, "days": days, "years": years})
    return pd.DataFrame(rows, columns=["expiration", "days", "years"])


# ---------- Black-76 ----------

# Undiscounted Black-76 option price. Works on numpy arrays. kind is "call" or "put".
def black_price(forward, strike, years, vol, kind="call"):
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * years) / (vol * np.sqrt(years))
    d2 = d1 - vol * np.sqrt(years)
    if kind == "call":
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


# Implied vol that makes black_price equal to `price`. Returns NaN if there is none.
def implied_vol(price, forward, strike, years, kind):
    def error(vol):
        return black_price(forward, strike, years, vol, kind) - price
    low, high = 0.001, 5.0
    if error(low) * error(high) > 0:  # same sign at both ends: no solution in range
        return np.nan
    return brentq(error, low, high)


# ---------- Forward price ----------

# Forward for one expiration from put-call parity: F = K + (C - P), using
# undiscounted mid prices at the strikes where call and put prices are closest.
def forward_from_parity(quotes):
    calls = quotes[quotes["type"] == "call"].set_index("strike")["mid"]
    puts = quotes[quotes["type"] == "put"].set_index("strike")["mid"]
    both = pd.DataFrame({"call": calls, "put": puts}).dropna()
    if both.empty:
        return np.nan
    both["gap"] = (both["call"] - both["put"]).abs()
    closest = both.nsmallest(3, "gap")  # the 3 strikes nearest the money
    estimates = closest.index + closest["call"] - closest["put"]
    return float(np.median(estimates))


# ---------- Cleaning ----------

# Mark quotes with a drop reason ("" = still kept). Returns a copy with
# columns mid and reason added. Prices become undiscounted (times exp(rT)).
def basic_filters(quotes, forward, years, rate, undiscounted_prices):
    quotes = quotes.copy()
    growth = 1.0 if undiscounted_prices else np.exp(rate * years)
    quotes["bid"] = quotes["bid"] * growth
    quotes["ask"] = quotes["ask"] * growth
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    quotes["reason"] = ""

    no_bid = quotes["bid"].isna() | (quotes["bid"] <= 0) | quotes["ask"].isna()
    quotes.loc[no_bid, "reason"] = "missing or zero bid"

    spread = (quotes["ask"] - quotes["bid"]) / quotes["mid"]
    wide = (quotes["reason"] == "") & (spread > MAX_SPREAD)
    quotes.loc[wide, "reason"] = "wide spread"

    low, high = STRIKE_BAND
    far = (quotes["strike"] < low * forward) | (quotes["strike"] > high * forward)
    quotes.loc[(quotes["reason"] == "") & far, "reason"] = "strike far from forward"
    return quotes


# Arbitrage filters for one option type. Prices must fall as we move away from
# the in-the-money side (calls: higher strikes, puts: lower strikes) and must be
# convex. A violation counts only if it is bigger than half the bid-ask spread.
def arbitrage_filters(quotes, kind):
    kept = quotes[(quotes["type"] == kind) & (quotes["reason"] == "")]
    # Order from in-the-money to out-of-the-money, so prices should go down
    kept = kept.sort_values("strike", ascending=(kind == "call"))
    tolerance = (quotes["ask"] - quotes["bid"]) / 2

    # 1) Decreasing: drop any quote that is not cheaper than the last kept one
    last_price = np.inf
    for index in kept.index:
        if kept.at[index, "mid"] > last_price + tolerance[index]:
            quotes.at[index, "reason"] = "price not decreasing (arbitrage)"
        else:
            last_price = kept.at[index, "mid"]

    # 2) Convex: each price must lie on or below the straight line between its
    #    neighbors. Drop the worst offender, then check again.
    kept = quotes[(quotes["type"] == kind) & (quotes["reason"] == "")].sort_values("strike")
    while len(kept) >= 3:
        strikes = kept["strike"].to_numpy()
        prices = kept["mid"].to_numpy()
        weight = (strikes[1:-1] - strikes[:-2]) / (strikes[2:] - strikes[:-2])
        line = prices[:-2] + weight * (prices[2:] - prices[:-2])  # neighbors' straight line
        excess = prices[1:-1] - line - tolerance[kept.index[1:-1]].to_numpy()
        worst = int(np.argmax(excess))
        if excess[worst] <= 0:
            break
        quotes.at[kept.index[worst + 1], "reason"] = "price not convex (arbitrage)"
        kept = kept.drop(kept.index[worst + 1])
    return quotes


# Compute our own implied vol for every quote that is still kept.
def add_implied_vols(quotes, forward, years):
    quotes["iv"] = np.nan
    for index in quotes.index[quotes["reason"] == ""]:
        row = quotes.loc[index]
        quotes.at[index, "iv"] = implied_vol(row["mid"], forward, row["strike"], years, row["type"])
    no_solution = (quotes["reason"] == "") & quotes["iv"].isna()
    quotes.loc[no_solution, "reason"] = "no IV solution"
    bad_iv = (quotes["reason"] == "") & ((quotes["iv"] < MIN_IV) | (quotes["iv"] > MAX_IV))
    quotes.loc[bad_iv, "reason"] = "IV outside 5%-300%"
    return quotes


# One IV per strike: puts below the forward, calls above, and a weighted
# blend inside +-BLEND_BAND (the call weight goes from 0 to 1 across the band).
def blend_smile(quotes, forward):
    good = quotes[quotes["reason"] == ""]
    calls = good[good["type"] == "call"].set_index("strike")["iv"]
    puts = good[good["type"] == "put"].set_index("strike")["iv"]
    low = forward * (1 - BLEND_BAND)
    high = forward * (1 + BLEND_BAND)

    points = []
    for strike in sorted(set(calls.index) | set(puts.index)):
        call_iv = calls.get(strike, np.nan)
        put_iv = puts.get(strike, np.nan)
        if strike < low:
            iv = put_iv
        elif strike > high:
            iv = call_iv
        elif np.isnan(call_iv) or np.isnan(put_iv):
            iv = put_iv if np.isnan(call_iv) else call_iv  # only one side quoted
        else:
            call_weight = (strike - low) / (high - low)
            iv = call_weight * call_iv + (1 - call_weight) * put_iv
        if not np.isnan(iv):
            points.append((strike, iv))
    return pd.DataFrame(points, columns=["strike", "iv"])


# ---------- Smoothing ----------

# Cubic smoothing spline of IV vs strike. s = n * noise^2 controls smoothness.
def fit_spline(strikes, ivs, strike_grid):
    spline = UnivariateSpline(strikes, ivs, k=3, s=len(strikes) * IV_NOISE ** 2)
    return spline(strike_grid)


# SVI total variance: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),
# where k = log(strike / forward). A popular 5-parameter shape for the smile.
def svi_variance(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# Fit SVI to the IV points (least squares on total variance) and return IVs on the grid.
def fit_svi(strikes, ivs, strike_grid, forward, years):
    k = np.log(strikes / forward)
    total_variance = ivs ** 2 * years

    def errors(params):
        return svi_variance(params, k) - total_variance

    start = [total_variance.min() * 0.5, 0.1, -0.3, 0.0, 0.1]   # a, b, rho, m, sigma
    lower = [0.0, 0.0, -0.999, -1.0, 0.001]
    upper = [total_variance.max(), 5.0, 0.999, 1.0, 2.0]
    fit = least_squares(errors, start, bounds=(lower, upper))
    grid_variance = svi_variance(fit.x, np.log(strike_grid / forward))
    return np.sqrt(np.clip(grid_variance, 1e-8, None) / years)


# ---------- Density and statistics ----------

# Breeden-Litzenberger: density = d²c/dK² (undiscounted call price), central differences.
# The first and last grid points are lost (they have no neighbor on one side).
def raw_density(strike_grid, call_prices):
    step = strike_grid[1] - strike_grid[0]
    second_diff = (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / step ** 2
    return strike_grid[1:-1], second_diff


# Price at a given cumulative probability (e.g. 0.05 for the 5th percentile).
def percentile_price(strikes, cdf, probability):
    return float(np.interp(probability, cdf, strikes))


# Probability that the price ends above `price`, from a result's CDF.
# Below the grid it is 1, above the grid it is 0 (no data out there).
def prob_above(result, price):
    return 1 - float(np.interp(price, result["strikes"], result["cdf"], left=0, right=1))


# Mean, standard deviation, skew, percentiles and tail probabilities.
def density_stats(strikes, density, cdf, spot):
    mean = trapezoid(strikes * density, strikes)
    std = np.sqrt(trapezoid((strikes - mean) ** 2 * density, strikes))
    skew = trapezoid(((strikes - mean) / std) ** 3 * density, strikes)
    return {
        "mean": mean, "std": std, "skew": skew,
        "p5": percentile_price(strikes, cdf, 0.05),
        "p50": percentile_price(strikes, cdf, 0.50),
        "p95": percentile_price(strikes, cdf, 0.95),
        "p_down_tail": float(np.interp(spot * (1 - TAIL_MOVE), strikes, cdf, left=0, right=1)),
        "p_up_tail": 1 - float(np.interp(spot * (1 + TAIL_MOVE), strikes, cdf, left=0, right=1)),
        "p_above_spot": 1 - float(np.interp(spot, strikes, cdf, left=0, right=1)),
    }


# ---------- One expiration ----------

# Full pipeline for one expiration. Returns a dictionary with the cleaned quotes,
# the smile, the fit, the density, statistics and a list of warnings (empty = all good).
def analyze_expiration(quotes, spot, forward, years, rate, method, undiscounted_prices):
    result = {"years": years, "days": years * 365, "forward": forward, "warnings": []}
    quotes = basic_filters(quotes, forward, years, rate, undiscounted_prices)
    quotes = arbitrage_filters(quotes, "call")
    quotes = arbitrage_filters(quotes, "put")
    quotes = add_implied_vols(quotes, forward, years)
    result["quotes"] = quotes

    smile = blend_smile(quotes, forward)
    result["smile"] = smile
    result["n_options"] = len(smile)
    if len(smile) < MIN_OPTIONS:
        result["warnings"].append(f"only {len(smile)} valid IV points (need {MIN_OPTIONS})")
        return result

    strikes = smile["strike"].to_numpy(dtype=float)
    ivs = smile["iv"].to_numpy(dtype=float)
    strike_grid = np.linspace(strikes.min(), strikes.max(), GRID_POINTS)
    if method == "SVI":
        iv_grid = fit_svi(strikes, ivs, strike_grid, forward, years)
    else:
        iv_grid = fit_spline(strikes, ivs, strike_grid)
    iv_grid = np.clip(iv_grid, MIN_IV, MAX_IV)  # keep the fitted curve in a sensible range
    call_prices = black_price(forward, strike_grid, years, iv_grid, "call")
    density_strikes, density = raw_density(strike_grid, call_prices)

    # Sanity numbers measured BEFORE cleaning the density
    positive_area = trapezoid(np.clip(density, 0, None), density_strikes)
    negative_area = -trapezoid(np.clip(density, None, 0), density_strikes)
    raw_mass = trapezoid(density, density_strikes)  # share of probability inside the strike range

    # Negative probabilities are impossible: set them to 0, then rescale to area 1
    density = np.clip(density, 0, None)
    density = density / trapezoid(density, density_strikes)
    cdf = cumulative_trapezoid(density, density_strikes, initial=0)
    peaks, _ = find_peaks(density, prominence=PEAK_PROMINENCE * density.max())

    result.update({
        "strike_grid": strike_grid, "iv_grid": iv_grid,
        "strikes": density_strikes, "density": density, "cdf": cdf,
        "raw_mass": raw_mass,
        "negative_share": negative_area / positive_area,
        "n_peaks": len(peaks),
    })
    result.update(density_stats(density_strikes, density, cdf, spot))
    result["mean_gap"] = result["mean"] / forward - 1
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


# ---------- Whole market ----------

# Forward price for one expiration: from the source (Deribit), from spot, or from parity.
def get_forward(market, quotes, years, rate):
    expiration = quotes["expiration"].iloc[0]
    if expiration in market["forwards"]:
        return market["forwards"][expiration]
    if market["spot"] is not None:
        return market["spot"] * np.exp(rate * years)  # no dividends
    growth = np.exp(rate * years)
    mids = quotes.assign(mid=(quotes["bid"] + quotes["ask"]) / 2 * growth)
    return forward_from_parity(mids[mids["bid"] > 0])


# Density on the shared surface grid (strike as % of spot), in probability per 1% of spot.
# Outside the strike range of the data the value is 0.
def density_on_surface_grid(result, spot):
    percent_of_spot = result["strikes"] / spot * 100
    density_per_percent = result["density"] * spot / 100  # per $1 -> per 1% of spot
    return np.interp(SURFACE_X, percent_of_spot, density_per_percent, left=0, right=0)


# Run the chosen expirations of a market. Returns spot, per-expiration results,
# and the surface built from the expirations that passed every check.
def analyze_market(market, expirations, rate, method):
    table = list_expirations(market)
    table = table[table["expiration"].isin(list(expirations))]
    spot = market["spot"]
    results = []
    for row in table.itertuples():
        quotes = market["quotes"][market["quotes"]["expiration"] == row.expiration]
        forward = get_forward(market, quotes, row.years, rate)
        if np.isnan(forward):
            continue  # CSV without spot and without matching call/put pairs
        if spot is None:  # CSV without a price column: spot = nearest forward, discounted
            spot = forward * np.exp(-rate * row.years)
        result = analyze_expiration(quotes, spot, forward, row.years, rate, method,
                                    market["undiscounted_prices"])
        result["expiration"] = row.expiration
        results.append(result)

    passed = [r for r in results if not r["warnings"]]
    surface_rows = [density_on_surface_grid(r, spot) for r in passed]
    return {"spot": spot, "results": results,
            "surface_days": [r["days"] for r in passed],
            "surface_z": np.array(surface_rows)}


# How many quotes were loaded and dropped (and why), per expiration plus a total row.
def quality_table(analysis):
    rows = []
    for r in analysis["results"]:
        reasons = r["quotes"]["reason"]
        row = {"Expiration": r["expiration"], "Loaded": len(reasons)}
        for reason in DROP_REASONS:
            row[reason] = int((reasons == reason).sum())
        row["Kept"] = int((reasons == "").sum())
        row["IV points used"] = r["n_options"]
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        total = table.drop(columns="Expiration").sum()
        total["Expiration"] = "Total"
        table = pd.concat([table, total.to_frame().T], ignore_index=True)
    return table


# One row per expiration with all summary statistics and the check result.
def summary_table(analysis):
    rows = []
    for r in analysis["results"]:
        row = {"expiration": r["expiration"], "days": round(r["days"], 1),
               "forward": r["forward"], "iv_points": r["n_options"]}
        if "density" in r:
            for key in ["mean", "mean_gap", "std", "skew", "p5", "p50", "p95",
                        "p_above_spot", "p_down_tail", "p_up_tail", "raw_mass"]:
                row[key] = r[key]
        row["checks"] = "OK" if not r["warnings"] else "; ".join(r["warnings"])
        rows.append(row)
    return pd.DataFrame(rows)


# All densities in long format (one row per expiration and strike), for download.
def density_table(analysis):
    parts = []
    for r in analysis["results"]:
        if "density" in r:
            parts.append(pd.DataFrame({"expiration": r["expiration"], "strike": r["strikes"],
                                       "density": r["density"], "cdf": r["cdf"]}))
    if not parts:
        return pd.DataFrame(columns=["expiration", "strike", "density", "cdf"])
    return pd.concat(parts, ignore_index=True)


# Command-line check: python implied_pdf.py
if __name__ == "__main__":
    import sources
    choice = input("Source: [1] NVDA yfinance  [2] BTC Deribit  [3] sample CSV (default 1): ").strip()
    if choice == "2":
        market = sources.load_deribit("BTC")
    elif choice == "3":
        with open(sources.SAMPLE_CSV, "rb") as file:
            market = sources.load_csv(file.read())
    else:
        market = sources.load_yfinance("NVDA")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for method in SMOOTHING_METHODS:
        analysis = analyze_market(market, list_expirations(market)["expiration"], DEFAULT_RATE, method)
        print(f"\n=== {market['name']} | spot {analysis['spot']:,.2f} | {method} | rate {DEFAULT_RATE:.0%} ===")
        print(quality_table(analysis).to_string(index=False))
        print(summary_table(analysis).drop(columns=["forward"]).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
        passed = len(analysis["surface_days"])
        print(f"{passed} of {len(analysis['results'])} expirations passed all checks.")
