import pandas as pd
import yfinance as yf
import math

# for terminal, signal beginning of script with time
print("\n", "Positions")
time1 = pd.Timestamp.now()
time1 = time1.floor(freq="S")
print("\n", time1)

# read in positions and history
df_positions = pd.read_csv("positions.csv")
df_history = pd.read_csv("history.csv")

# get the beginning balance from the trade history
balance = df_history.iloc[0]["BALANCE"]

# only take enough positions to have three
# the count is how many to buy in this iteration
count = 3 - len(df_positions)


# is there room to take new position?
# max number of positions is three
if count == 0:
    exit("The previous three positions are still being held.")

# read in the top pics for trading
df_buys = pd.read_csv("scan.csv", nrows=count)
if len(df_buys) == 0:
    exit("Today's not the day to be buying stocks!")

# get list of tickers to buy from the 'scan' dataset
buys = df_buys["STOCK"].tolist()

# retrieve the stock information from yahoo!
# and keep only the closing prices
df = yf.download(buys, group_by="Ticker", period="1d")
df = df.stack(level=0).rename_axis(["Date", "STOCK"]).reset_index(level=1)
if len(buys) > 1:
    not_needed = ["High", "Low", "Adj Close", "Open", "Volume"]
    df = df.drop(not_needed, axis=1)
    df.reset_index(inplace=True)
else:
    df.reset_index(inplace=True)
    df = df.loc[[4]]
    df.iat[0, 1] = buys[0]
    df.columns = ["Date", "STOCK", "Close"]

# merge the "scan" dataset with the new df
# and sort so that least expensive stock is last
# to maximize dollars spent in trades
df_1 = pd.merge(df, df_buys, on="STOCK")
df_1 = df_1.sort_values(["Close"], ascending=False, ignore_index=True)

# loop through each new purchase to add items to the
# 'positions' dataset, one new row (nuro) at a time
# the allocation will be the dollar amount per position to purchase
# Timedeltas used to create date for sale conditions
allocation = 0
expiration_1 = pd.Timedelta(1, "D")
expiration_2 = pd.Timedelta(2, "D")
for i in range(count):
    allocation = balance / (count - i)
    stock = df_1.iloc[i]["STOCK"]
    buy_date = pd.Timestamp.now()
    sell_date_1 = buy_date + expiration_1
    sell_date_2 = buy_date + expiration_2
    buy_date = buy_date.floor(freq="T")
    sell_date_1 = sell_date_1.floor(freq="T")
    sell_date_2 = sell_date_2.floor(freq="T")
    share_cost = (df_1.iloc[i]["Close"]).round(3)
    num_shares = math.floor(allocation / share_cost)
    total_cost = (share_cost * num_shares).round(2)
    exit_trigger = (df_1.iloc[i]["SellAt"]).round(3)
    exit_hit = (df_1.iloc[i]["HitAt"]).round(3)
    balance = balance - total_cost
    nuro = [stock,
            buy_date,
            share_cost,
            num_shares,
            total_cost,
            exit_trigger,
            exit_hit,
            sell_date_1,
            sell_date_2,
            balance]
    df_2 = pd.DataFrame(
        [nuro],
        columns=[
            "STOCK",
            "Buy Date",
            "Share Price",
            "Shares",
            "Cost",
            "Target Price",
            "Hit Price",
            "Sell On 1",
            "Sell On 2",
            "BALANCE",
        ],
    )
    # add the new rows to the existing positions dataset
    df_positions = pd.concat([df_positions, df_2], ignore_index=True)
    nuro.clear()

# normalize the dates so that they are properly formatted for future use
df_positions['Buy Date'] = pd.DatetimeIndex(df_positions['Buy Date']).normalize()
df_positions['Sell On 1'] = pd.DatetimeIndex(df_positions['Sell On 1']).normalize()
df_positions['Sell On 2'] = pd.DatetimeIndex(df_positions['Sell On 2']).normalize()

# save to csv for future use
df_positions.to_csv("positions.csv", index=False)

# print to terminal
print("\nPositions", df_positions)

# print end time info to terminal
time = pd.Timestamp.now()
time = time.floor(freq="S")
elapsed = time - time1
print("\n", time, "\n", elapsed, "\n")
