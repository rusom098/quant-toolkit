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
def get_equity(returns, amount=10000):
    return amount * (1 + returns).cumprod()

def get_annual_return(equity):
    years = len(equity) / 252
    return (equity.iloc[-1] / equity.iloc[0] ** (1 / years) - 1)

def get_volatility(returns):
    return returns.std() * 252 ** 0.5

def get_sharpe(returns):
    return returns.mean() / returns.std() * 252 ** 0.5