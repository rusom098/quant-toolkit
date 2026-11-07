import matplotlib.pyplot as plt
from data import get_prices, get_returns, get_growth

ticker = input("Enter a ticker: ")
prices = get_prices(ticker, "2020-01-01", "2025-01-01")
print(prices.head())
print(len(prices), "trading days")

returns = get_returns(prices)


print(returns.head())

print("Best day:", returns.idxmax().date(), f"{returns.max():.2%}")
print("Worst day:", returns.idxmin().date(), f"{returns.min():.2%}")


growth = get_growth(prices, 10000)
print(f"$10,000 in 2020 grew to ${growth:,.2f}")
growth = get_growth(prices, 5000)
print(f"$5000 in 2020 grew to ${growth:,.2f}")

equity = 10000 * (prices / prices.iloc[0])
print(equity.head())
print(equity.tail())

bg = "#262624"
text = "#C2C0B6"
line = "#A8C5DA"

plt.style.use("dark_background")
plt.figure(figsize=(12,6), facecolor=bg)
plt.gca().set_facecolor(bg)

plt.plot(equity, color=line, linewidth=1.5)
plt.title(f"Growth of $10,000 Invested in {ticker} (2020–2024)", color=text)
plt.xlabel("Date", color=text)
plt.ylabel("Portfolio Value ($)", color=text)
plt.tick_params(color=text)
plt.grid(True, color=text, alpha=0.1)

for side in["top", "right"]:
    plt.gca().spines[side].set_visible(False)

plt.gca().yaxis.set_major_formatter("${x:,.0f}")
print("Worst day: ", returns.idxmin().date(), f"{returns.min():.2%}")

plt.show()

