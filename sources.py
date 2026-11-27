"""
Option data sources. Each loader returns the same "market" dictionary:

    name                 text shown on the page, e.g. "NVDA (yfinance)"
    spot                 current underlying price, or None if unknown (CSV)
    spot_note            where the spot price came from
    as_of                time the quotes are from (UTC datetime)
    quotes               table: expiration, expiry_time, type, strike, bid, ask
    forwards             {expiration: forward price} if the source gives one
    undiscounted_prices  True if prices are already "forward" values (Deribit)

implied_pdf.py only works with this dictionary, so it does not care where
the data came from.
"""
import datetime
import io
import os

import pandas as pd
import requests
import yfinance as yf

from data import get_prices
from implied_pdf import MIN_DAYS, MAX_DAYS, years_between

DERIBIT_URL = "https://www.deribit.com/api/v2/public/"
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "samples", "sample_options_SPY_2026-11-25.csv")
US_CLOSE_UTC = 21   # US options expire at the 4 pm New York close (21:00 UTC in winter)
DERIBIT_EXPIRY_UTC = 8  # Deribit options expire at 08:00 UTC
QUOTE_COLUMNS = ["expiration", "expiry_time", "type", "strike", "bid", "ask"]


# Current time in UTC (time-zone aware).
def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


# Expiry moment for a date like "2026-12-18" at the given UTC hour.
def expiry_moment(date_text, hour):
    date = datetime.date.fromisoformat(date_text)
    return datetime.datetime(date.year, date.month, date.day, hour, tzinfo=datetime.timezone.utc)


# True if an expiry is between MIN_DAYS and MAX_DAYS after `as_of`.
def in_window(as_of, expiry_time):
    days = years_between(as_of, expiry_time) * 365
    return MIN_DAYS <= days <= MAX_DAYS


# ---------- 1) yfinance (US stocks and ETFs) ----------

# Latest close, reusing get_prices from data.py.
def get_spot(ticker):
    today = datetime.date.today()
    start = str(today - datetime.timedelta(days=10))  # 10 days back covers weekends/holidays
    end = str(today + datetime.timedelta(days=1))     # end date is exclusive in yfinance
    prices = get_prices(ticker, start, end).dropna()
    return float(prices.iloc[-1]), prices.index[-1].date()


# Option chains from Yahoo Finance for expirations in the 7-90 day window.
def load_yfinance(ticker):
    spot, spot_date = get_spot(ticker)
    as_of = now_utc()
    stock = yf.Ticker(ticker)
    tables = []
    for expiration in stock.options:
        expiry_time = expiry_moment(expiration, US_CLOSE_UTC)
        if not in_window(as_of, expiry_time):
            continue  # skip downloads we would not use
        chain = stock.option_chain(expiration)
        for kind, table in [("call", chain.calls), ("put", chain.puts)]:
            tables.append(pd.DataFrame({
                "expiration": expiration, "expiry_time": expiry_time, "type": kind,
                "strike": table["strike"], "bid": table["bid"], "ask": table["ask"]}))
    if not tables:
        raise ValueError(f"No {ticker} option expirations between {MIN_DAYS} and {MAX_DAYS} days.")
    return {
        "name": f"{ticker} (yfinance)", "spot": spot,
        "spot_note": f"last close on {spot_date}; if the market is closed, option quotes "
                     "are from the last trading session",
        "as_of": as_of, "quotes": pd.concat(tables, ignore_index=True)[QUOTE_COLUMNS],
        "forwards": {}, "undiscounted_prices": False,
    }


# ---------- 2) Deribit (BTC and ETH, public API, no key) ----------

# Call one public Deribit endpoint and return its "result" part.
def deribit_get(method, params):
    response = requests.get(DERIBIT_URL + method, params=params, timeout=20)
    response.raise_for_status()
    return response.json()["result"]


# Turn "27DEC26" into "2026-12-27".
def deribit_date(text):
    return datetime.datetime.strptime(text, "%d%b%y").date().isoformat()


# Live BTC or ETH option quotes from Deribit, converted to USD.
# Deribit quotes prices in BTC (or ETH). Price in coins * the forward price
# (`underlying_price`) is the undiscounted USD value, so no extra discounting.
def load_deribit(currency):
    summaries = deribit_get("get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    index = deribit_get("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    as_of = now_utc()

    rows = []
    for item in summaries:
        _, date_code, strike, letter = item["instrument_name"].split("-")  # BTC-27DEC26-100000-C
        expiration = deribit_date(date_code)
        expiry_time = expiry_moment(expiration, DERIBIT_EXPIRY_UTC)
        if not in_window(as_of, expiry_time):
            continue
        forward = item["underlying_price"]
        bid = item["bid_price"]  # None when nobody is bidding
        ask = item["ask_price"]
        rows.append({
            "expiration": expiration, "expiry_time": expiry_time,
            "type": "call" if letter == "C" else "put", "strike": float(strike),
            "bid": bid * forward if bid else float("nan"),
            "ask": ask * forward if ask else float("nan"),
            "forward": forward,
        })
    if not rows:
        raise ValueError(f"No {currency} option expirations between {MIN_DAYS} and {MAX_DAYS} days.")
    quotes = pd.DataFrame(rows)
    forwards = quotes.groupby("expiration")["forward"].median().to_dict()
    return {
        "name": f"{currency} (Deribit)", "spot": float(index["index_price"]),
        "spot_note": f"Deribit {currency.lower()}_usd index, live at "
                     f"{as_of:%Y-%m-%d %H:%M} UTC; forwards from Deribit futures",
        "as_of": as_of, "quotes": quotes[QUOTE_COLUMNS],
        "forwards": forwards, "undiscounted_prices": True,
    }


# ---------- 3) CSV upload ----------

# Check an uploaded CSV and return a list of problems (empty list = file is fine).
def validate_csv(table):
    required = ["strike", "expiration", "type", "bid", "ask"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        return [f"Missing column(s): {', '.join(missing)}. Required: {', '.join(required)}."]
    if table.empty:
        return ["The file has no rows."]

    problems = []
    for column in ["strike", "bid", "ask"]:
        numbers = pd.to_numeric(table[column], errors="coerce")
        bad_rows = table.index[numbers.isna() & table[column].notna()]
        if len(bad_rows):
            problems.append(f"'{column}' is not a number in row(s) {row_list(bad_rows)}.")
    strikes = pd.to_numeric(table["strike"], errors="coerce")
    if (strikes <= 0).any():
        problems.append(f"'strike' must be positive (row(s) {row_list(table.index[strikes <= 0])}).")

    dates = pd.to_datetime(table["expiration"], errors="coerce", format="%Y-%m-%d")
    if dates.isna().any():
        problems.append(f"'expiration' must look like 2026-12-18 (row(s) {row_list(table.index[dates.isna()])}).")

    types = table["type"].astype(str).str.strip().str.lower()
    bad_type = ~types.isin(["call", "put", "c", "p"])
    if bad_type.any():
        problems.append(f"'type' must be call or put (row(s) {row_list(table.index[bad_type])}).")

    bids = pd.to_numeric(table["bid"], errors="coerce")
    asks = pd.to_numeric(table["ask"], errors="coerce")
    if (bids < 0).any() or (asks < 0).any():
        problems.append("'bid' and 'ask' cannot be negative.")
    crossed = bids > asks
    if crossed.any():
        problems.append(f"'bid' is above 'ask' in row(s) {row_list(table.index[crossed])}.")

    if "underlying_price" in table.columns:
        prices = pd.to_numeric(table["underlying_price"], errors="coerce").dropna()
        if (prices <= 0).any():
            problems.append("'underlying_price' must be positive.")
    if "quote_date" in table.columns:
        quote_dates = pd.to_datetime(table["quote_date"], errors="coerce", format="%Y-%m-%d")
        if quote_dates.isna().any() or quote_dates.nunique() > 1:
            problems.append("'quote_date' must be one date like 2026-11-25 for the whole file.")
    return problems


# Turn row positions into readable row numbers (header = row 1), at most 5 shown.
def row_list(index):
    numbers = [str(i + 2) for i in list(index)[:5]]
    more = " ..." if len(index) > 5 else ""
    return ", ".join(numbers) + more


# Read uploaded CSV bytes into a market dictionary. Raises ValueError with
# clear messages if the file is not valid.
# Optional columns: underlying_price (spot), quote_date (when the quotes were taken).
def load_csv(file_bytes):
    try:
        table = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as error:
        raise ValueError(f"Could not read the file as CSV: {error}")
    table.columns = [column.strip().lower() for column in table.columns]
    problems = validate_csv(table)
    if problems:
        raise ValueError(" ".join(problems))

    if "quote_date" in table.columns:
        as_of = expiry_moment(str(table["quote_date"].iloc[0])[:10], US_CLOSE_UTC)
        as_of_note = f"quotes dated {as_of.date()} (from the file)"
    else:
        as_of = now_utc()
        as_of_note = "no quote_date column, so time to expiry is measured from now"

    spot = None
    spot_note = "no underlying_price column: forward inferred from put-call parity"
    if "underlying_price" in table.columns and table["underlying_price"].notna().any():
        spot = float(pd.to_numeric(table["underlying_price"], errors="coerce").dropna().iloc[0])
        spot_note = "underlying_price column"

    types = table["type"].astype(str).str.strip().str.lower().replace({"c": "call", "p": "put"})
    expirations = pd.to_datetime(table["expiration"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    quotes = pd.DataFrame({
        "expiration": expirations,
        "expiry_time": [expiry_moment(text, US_CLOSE_UTC) for text in expirations],
        "type": types,
        "strike": pd.to_numeric(table["strike"]),
        "bid": pd.to_numeric(table["bid"]),
        "ask": pd.to_numeric(table["ask"]),
    })
    in_range = [in_window(as_of, moment) for moment in quotes["expiry_time"]]
    if not any(in_range):
        raise ValueError(f"No expirations are {MIN_DAYS}-{MAX_DAYS} days after the quote date "
                         f"({as_of.date()}). Add a quote_date column or use newer quotes.")
    return {
        "name": "Uploaded CSV", "spot": spot, "spot_note": f"{spot_note}; {as_of_note}",
        "as_of": as_of, "quotes": quotes, "forwards": {}, "undiscounted_prices": False,
    }
