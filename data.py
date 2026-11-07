import yfinance as yf

def get_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"][ticker]
    return prices

def get_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def get_growth(prices, amount):
    growth = amount * (prices.iloc[-1] / prices.iloc[0])
    return growth

def get_max_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity-peak)/peak
    worst = drawdown.min()
    return worst

