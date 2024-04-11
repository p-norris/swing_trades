# scan - 9
# positions - 299
# exits - 433
# interface - 716
# app - 1980

import dash_mantine_components as dmc

scan_file = dmc.Prism('''
"""
    After the tickers.py file downloads a year's worth of price data for
    over 6,000 stocks, it then filters out stocks that do not meet minimum 
    criteria. Then this scan.py file uses those tickers to generate moving
    averages data and buy signals. It also determines price levels for 
    selling stocks. The data is sorted according to a few metrics and then
    saved for the positions.py file to use in 'implementing' the trades.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time


def lets_go():
    time = pd.Timestamp.now()
    time = time.floor(freq="s")
    return time


def considerations():
    Tickers = pd.read_csv("src/tickers.csv")
    tickers = Tickers["SYMBOL"].to_list()
    stocks = ticker_histories(tickers, history="282d")
    return stocks, tickers


def the_dataframe():
    final = pd.DataFrame(
        columns=[
            "STOCK",
            "P",
            "sig",
            "BUY",
            "Activity",
            "Strength",
            "HIGH",
            "CLOSE",
            "Multiplier",
            "SellAt",
            "HitAt",
        ]
    )
    return final


def ticker_histories(tickers, history):
    stocks = yf.download(tickers, group_by="ticker", period=history)
    return stocks


def what_stock(ticker):
    nuro = [ticker]
    return nuro


def rolling_MAs(df):
    ma_3 = df["Close"].rolling(3).mean().round(3)
    ma_6 = df["Close"].rolling(6).mean().round(3)
    ma_13 = df["Close"].rolling(13).mean().round(3)
    ma_21 = df["Close"].rolling(21).mean().round(3)
    df = df.assign(MA3=ma_3, MA6=ma_6, MA13=ma_13, MA21=ma_21)
    return df


def testing(df):
    df["Target"] = (df["Close"]) * 1.011
    df["Max1"] = df["High"].shift(-1).round(2)
    df["Max2"] = df["High"].shift(-2).round(2)
    df["Trigger"] = np.where(
        (
                (df["Close"] >= df["MA3"])
                & (df["MA3"] >= df["MA6"])
                & (df["MA3"] >= df["MA13"])
                & (df["MA3"] >= df["MA21"])
        ),
        np.where((df["Target"] < df["Max1"]), "Gain", "Loss"),
        0,
    )
    return df


def p(df, nuro):
    px = df["Trigger"].to_numpy()
    pxx = px[:-1]
    G009 = np.count_nonzero(pxx == "Gain")
    L009 = np.count_nonzero(pxx == "Loss")
    if G009 | L009 == 0:
        return -1, -1, -1
    p_tally = round(G009 / (G009 + L009), 3)
    if p_tally < 0.8:
        return -1, -1, -1
    nuro.append(p_tally)
    return p_tally, px, nuro


def signs(Px009, nuro):
    six_off = Px009[-6]
    five_off = Px009[-5]
    four_from_last = Px009[-4]
    antepenultimate = Px009[-3]
    penultimate = Px009[-2]
    ultimate = Px009[-1]
    if (
            (six_off == "Gain")
            & (five_off == "0")
            & (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "0")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "6"
    if (
            (five_off == "Gain")
            & (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "0")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "5"
    if (
            (four_from_last == "Gain")
            & (antepenultimate == "0")
            & (penultimate == "0")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "4"
    if (
            (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "0")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "1"
    elif (
            (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "Gain")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "2"
    elif (
            (four_from_last == "0")
            & (antepenultimate == "Gain")
            & (penultimate == "Gain")
            & (ultimate == "Loss")
    ):
        buy, sig = "Yes", "3"
    else:
        buy, sig = "No", "0"
    nuro.append(sig)
    nuro.append(buy)
    return nuro


def occurrences(px, nuro):
    zeros = np.count_nonzero(px == "0")
    array_length = len(px)
    activity = 1 - (zeros / (array_length))
    if activity < 0.1999:
        return -1, -1
    nuro.append(round(activity, 3))
    return activity, nuro


def strength(activity, p_tally, nuro):
    strength = round(
        ((p_tally * p_tally * p_tally * activity) * 396.5), 1
    )
    nuro.append(strength)
    return nuro


def high_close(df, nuro):
    h = df["High"].astype(float)
    high = h.iloc[-1]
    c = df["Close"].astype(float)
    close = c.iloc[-1]
    nuro.append(round(high, 2))
    nuro.append(round(close, 2))
    return nuro


def safe_gain(df, px, nuro):
    target = []
    pxClose = df["Close"].astype(float)
    pxHigh = df["High"].astype(float)

    for i in range(len(px) - 1):
        if px[i] == "Gain":
            x = (pxHigh.iloc[i + 1] - pxClose.iloc[i]) / pxHigh.iloc[i + 1]
            target.append(x)
    target.sort()

    if len(target) > 5:
        multiplier = 1 + np.quantile(target, 0.375)
        sell_price = pxClose.iloc[-1] * multiplier
    else:
        sell_price = pxClose.iloc[-1] * 1.011
    nuro.append(round(multiplier - 1, 3))
    nuro.append(round(sell_price, 2))
    return pxClose, nuro


def take_the_hit(nuro):
    hit_pct = -0.025
    take_hit = pxClose.iloc[-1] * (1 + hit_pct)
    nuro.append(round(take_hit, 2))
    return nuro


def update_scan_dataset(nuro):
    final.loc[len(final)] = nuro
    return final


def being_picky(final):
    if len(final) > 0:
        final = final.sort_values(
            ["BUY", "Strength", "P"], ascending=[False, False, False]
        )
        final = final[final["BUY"] == "Yes"]
        return final


def an_ending(final):
    final = final.reset_index(drop=True)
    final.to_csv("src/scan.csv", index=False)
    return final


# MAIN ***********************************************************

if __name__ == "__main__":
    print("\\n", "Today's Stocks")
    time1 = lets_go()
    print("\\n", time1)
    final = the_dataframe()
    stocks, tickers = considerations()

    for ticker in tickers:
        df = pd.DataFrame(stocks[ticker].to_records())

        nuro = what_stock(ticker)

        df = rolling_MAs(df)

        df = testing(df)

        p_tally, px, nuro = p(df, nuro)
        if p_tally == -1:
            continue

        nuro = signs(px, nuro)

        activity, nuro = occurrences(px, nuro)
        if activity == -1:
            continue

        nuro = strength(activity, p_tally, nuro)

        nuro = high_close(df, nuro)

        pxClose, nuro = safe_gain(df, px, nuro)

        nuro = take_the_hit(nuro)

        final = update_scan_dataset(nuro)

    ah_finally = being_picky(final)

    final = an_ending(ah_finally)

    print('\\nSTRENGTH: Superior: 70-100, Excellent: 56-69, Very Good: 46-55, Good: 40-45')
    print(final)

    time = pd.Timestamp.now()
    time = time.floor(freq="s")
    elapsed = (time - time1)
    elapsed = elapsed.total_seconds()
    print("\\n", time, "\\n", elapsed, "seconds\\n")


''',
                      language="python",
                      withLineNumbers=True,
                      )

positions_file = dmc.Prism('''
"""
    The positions.py file takes the scan.py information and
    uses it to 'make purchases' of stocks that best fit this
    trading strategy. It takes into account positions that it
    is currently holding as well as the possibility that the
    scan did not find enough good buys for the current day. 
"""

import pandas as pd
import yfinance as yf
import math


def time_now():
    right_now = pd.Timestamp.now().strftime("%H:%M:%S")
    return right_now


def the_date():
    date_today = str(pd.Timestamp.now().date())
    return date_today


def get_balance():
    history = pd.read_csv("src/history.csv")
    temp_balance = history.iloc[0]["BALANCE"]
    positions = pd.read_csv("src/positions.csv")
    if len(positions) > 0:
        outstanding = positions["Cost"].sum()
    else:
        outstanding = 0
    balance = (temp_balance - outstanding).round(2)
    return balance


def price(position):
    df = yf.download(position, period="1d")
    price = df.iloc[0, 3]
    return price


def num_holding():
    positions = pd.read_csv("src/positions.csv")
    holding_num = len(positions)
    if holding_num == 3:
        exit("\\nThe previous position(s) are still being held.\\n")
    return holding_num


def dont_buy_these():
    scan = pd.read_csv("src/scan.csv")
    positions = pd.read_csv("src/positions.csv")
    stocks = list(positions.STOCK)
    for stock in stocks:
        scan = scan[scan["STOCK"] != stock]
    return scan


def buy_these(scan, buy_number):
    scan = scan.head(buy_number)
    if len(scan) == 0:
        quit("\\nNothing to buy.\\n")
    scan = scan.sort_values(["CLOSE"], ascending=False)
    buy_list = list(scan["STOCK"])
    return buy_list


def sell_date():
    hours = pd.read_csv("src/hours.csv")
    sell_on = str(pd.to_datetime(hours.loc[2, "date"]).date())
    return sell_on


def build_row(ticker, share_cost, balance):
    scan = pd.read_csv("src/scan.csv")
    multiplier = scan.loc[scan.STOCK == ticker, "Multiplier"].squeeze()
    num_shares = math.floor(balance / share_cost)
    close = scan.loc[scan.STOCK == ticker, "CLOSE"].squeeze()
    date = the_date()
    sell_on = sell_date()
    nuro = []
    nuro.append(ticker)
    nuro.append(date)
    nuro.append(sell_on)
    nuro.append(scan.loc[scan.STOCK == ticker, "sig"].squeeze())
    nuro.append((scan.loc[scan.STOCK == ticker, "P"].squeeze()).round(3))
    nuro.append((scan.loc[scan.STOCK == ticker, "Activity"].squeeze()).round(3))
    nuro.append((scan.loc[scan.STOCK == ticker, "Strength"].squeeze()).round(3))
    nuro.append(multiplier.round(3))
    nuro.append((scan.loc[scan.STOCK == ticker, "HIGH"].squeeze()).round(3))
    nuro.append(share_cost.round(3))
    nuro.append(num_shares)
    nuro.append((num_shares * share_cost).round(3))  # position cost
    nuro.append((share_cost * (1 + multiplier)).round(3))  # target price
    nuro.append((close * 0.94).round(3))
    nuro.append((close * 0.975).round(3))
    return nuro


def update_positions(nuro):
    positions = pd.read_csv("src/positions.csv")
    positions.loc[len(positions)] = nuro
    positions.to_csv("src/positions.csv", index=False)
    return positions


def buy_stock(buy_number, buy_list):
    for i in range(0, buy_number):
        bal = get_balance()
        balance = bal / buy_number
        ticker = buy_list[i]
        share_cost = price(ticker)
        nuro = build_row(ticker, share_cost, balance)
        positions = update_positions(nuro)
    return positions


if __name__ == "__main__":
    print('Positions:')
    current_number = num_holding()
    buy_number = 3 - current_number
    scan = dont_buy_these()
    buy_list = buy_these(scan, buy_number)
    positions = buy_stock(buy_number, buy_list)
    print("\\nPositions",positions, "\\n")
    right_now = time_now()
    print("\\nPositions -", right_now, "\\n")

''',
                           language="python",
                           withLineNumbers=True,
                           )

exits_file = dmc.Prism('''
"""
    The exits.py file sells the held position when an exit
    criteria is met. Stocks are always sold the day after
    purchase or the day after that. On the first of these
    two days, the exit trigger for a loss is suspended so that
    a high value (profitable) exit has enough time to occur.
    However, on the second day, if the high exit does not occur
    within the first 30 minutes of trading, the low value exit
    (for a loss) goes into effect. If neither occur, the position
    is sold at the end of this second day for market value.
"""

import pandas as pd
import yfinance as yf
import time


def begin():
    print("\\n", "sellSchwab")
    time1 = pd.Timestamp.now()
    time1 = time1.floor(freq="s")
    print("\\n", time1)


def get_times():
    hours = pd.read_csv("src/hours.csv")
    get_hit = hours.loc[0, "at_10_10"]
    two_til_four = hours.loc[0, "at_two_til"]
    one_til_four = hours.loc[0, "at_one_til"]
    return get_hit, two_til_four, one_til_four


def get_positions():
    positions = pd.read_csv('src/positions.csv')
    if len(positions) == 0:
        exit("\nThere's nothing to trade.\\n")
    return positions


def get_sectors():
    df_all_sectors = pd.read_csv("src/tickers.csv")
    sectors_dict = dict(
        zip(
            df_all_sectors.SYMBOL,
            df_all_sectors.SECTOR,
        )
    )
    return sectors_dict


def time_now():
    right_now = pd.Timestamp.now().strftime("%H:%M")
    return right_now


def the_date():
    date_today = str(pd.Timestamp.now().date())
    return date_today


def data_list():
    positions = get_positions()
    collected_data = []
    for i in range(len(positions)):
        data_list = [True, positions.loc[i, "STOCK"], positions.loc[i, "Shares"], positions.loc[i, "Target Price"],
                     positions.loc[i, "Hit_94%"], positions.loc[i, "Hit_97.5%"], positions.loc[i, "Sell On"]]
        collected_data.append(data_list)
    return collected_data


def price(position):
    try:
        df = yf.download(position, period="1d")
        price = df.iloc[0, 3]
    except:
        print(position, 'failed to download properly.\\n')
        price = -1
    return price


def get_balance():
    history = pd.read_csv("src/history.csv")
    balance = history.iloc[0]["BALANCE"]
    return balance


def update_positions(nuro):
    positions = get_positions()
    positions.loc[len(positions)] = nuro
    positions.to_csv("src/positions.csv", index=False)
    return positions


def trade_record(ticker, sell_price):
    positions = get_positions()
    sectors_dict = get_sectors()
    buy_date = positions.loc[positions["STOCK"] == ticker, "Buy Date"].squeeze()
    sell_date = str(pd.Timestamp.now().date())
    share_price = (positions.loc[positions["STOCK"] == ticker, "Share Price"].squeeze()).round(3)
    shares = (positions.loc[positions["STOCK"] == ticker, "Shares"].squeeze())
    cost = round(shares * share_price, 2)
    sell_price = round(sell_price, 2)
    income = round(shares * sell_price, 2)
    profit = round(income - cost, 2)
    pct_chg = round((profit / cost) * 100, 2)
    prev_bal = get_balance()
    balance = round(prev_bal + profit, 2)
    nuro = [
        ticker,
        sectors_dict[ticker],
        buy_date,
        sell_date,
        shares,
        share_price,
        cost,
        sell_price,
        income,
        profit,
        pct_chg,
        balance,
    ]
    return nuro


def update_datasets(ticker, nuro):
    history = pd.read_csv("src/history.csv")
    history.loc[-1] = nuro
    history.index = history.index + 1
    history.sort_index(inplace=True)
    history.to_csv("src/history.csv", index=False)

    nuro.clear()

    positions = get_positions()
    positions = positions[positions["STOCK"] != ticker]
    positions = positions.reset_index(drop=True)
    positions.to_csv("src/positions.csv", index=False)

    num_positions = len(positions)
    if num_positions == 0:
        print("\\n\nDONE!\nHISTORY:\\n")
        print(history.head(), "\n")
        return False
    return True


def go_for_a_run(ticker):
    print(ticker, "is going on a run!\\n")
    hours = pd.read_csv("src/hours.csv")
    positions = get_positions()
    row_num = positions.loc[positions.STOCK == ticker].index[0]
    new_date = str(pd.to_datetime(hours.loc[1, "date"]).date())
    positions.at[row_num, "Sell On"] = new_date
    positions.to_csv("src/positions.csv", index=False)

    scan = pd.read_csv("src/scan.csv")
    scan = scan[scan["STOCK"] != ticker]
    scan.to_csv("src/scan.csv", index=False)
    return


def sell_as_is(ticker):
    print(ticker, ": FOR SALE -- AS IS\\n")
    quote = price(ticker)
    if quote == -1: print('\\n\\nERROR!\\n\\n')
    nuro = trade_record(ticker, quote)
    holding = update_datasets(ticker, nuro)
    return holding


def get_money(stock_data, cashed_tickers, get_hit):
    holding = True
    while holding:
        for i in range(len(stock_data)):
            this_stock = stock_data[i]
            ticker = this_stock[1]
            if ticker in cashed_tickers:
                continue
            gain = this_stock[3]
            right_now = time_now()
            if right_now == get_hit:
                holding = False
                break
            quote = price(ticker)
            if quote == -1: 
                continue
            print("g_m_QUOTE at", time_now() + ':', ticker + ":", quote, "vs", gain, "<- target\n")
            if quote >= gain:
                nuro = trade_record(ticker, quote)
                holding = update_datasets(ticker, nuro)
                cashed_tickers.append(ticker)
                print(ticker, '- Sold Early\\n')
                continue
        time.sleep(31)
    return cashed_tickers, holding


def get_more_money(stock_data, cashed_tickers, holding, two_til_four, one_til_four):
    holding = True
    date = the_date()
    while holding:
        for i in range(len(stock_data)):
            this_stock = stock_data[i]
            ticker = this_stock[1]
            if ticker in cashed_tickers:
                continue
            gain = this_stock[3]
            loss1 = this_stock[4]
            loss2 = this_stock[5]
            exit_date = this_stock[6]
            quote = price(ticker)
            if quote == -1:
                continue
            right_now = time_now()
            print("g_m_m_QUOTE at", right_now + ':', ticker + ":", quote, "vs", gain, "<- target price\n")
            if quote >= gain:
                nuro = trade_record(ticker, quote)
                holding = update_datasets(ticker, nuro)
                cashed_tickers.append(ticker)
                print(ticker, '- Sold High\\n')
                continue
            elif date != exit_date:
                if quote < loss1:
                    nuro = trade_record(ticker, quote)
                    holding = update_datasets(ticker, nuro)
                    cashed_tickers.append(ticker)
                    print(ticker, '- Sold Low, 94%\\n')
                    continue
            elif date == exit_date:
                if quote < loss2:
                    nuro = trade_record(ticker, quote)
                    holding = update_datasets(ticker, nuro)
                    cashed_tickers.append(ticker)
                    print(ticker, '- Sold Low, 97.5%\\n')
                    continue
                elif right_now == two_til_four:
                    scan = pd.read_csv("src/scan.csv")
                    new_tickers = scan["STOCK"].tolist()
                    if ticker in new_tickers:
                        go_for_a_run(ticker)
                        cashed_tickers.append(ticker)
                        continue
                    else:
                        holding = sell_as_is(ticker)
                        cashed_tickers.append(ticker)
                        continue
            if right_now == one_til_four:
                holding = False
                break
            if len(stock_data) == len(cashed_tickers):
                holding = False
                break
        time.sleep(31)
    return


def sell_stock(stock_data): # MAIN()
    cashed_tickers = []
    get_hit, two_til_four, one_til_four = get_times()

    print('\\n\\n\\n**** PHASE ONE: GET MONEY!!! ****\\n\\n\\n')

    cashed_tickers, holding = get_money(stock_data, cashed_tickers, get_hit)
    if len(stock_data) == len(cashed_tickers):
        holding = False

    print('\\n\\n\\n**** PHASE TWO: GET MORE MONEY!!! ****\\n\\n\\n')

    get_more_money(stock_data, cashed_tickers, holding, two_til_four, one_til_four)
    return


if __name__ == "__main__":
    begin()
    stock_data = data_list()
    sell_stock(stock_data)


''',
                       language="python",
                       withLineNumbers=True,
                       )

interface_file = dmc.Prism('''
"""
    Here is the code that produces the dashboard
    you are currently viewing.
"""

import pandas as pd
import numpy as np
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression

from dash import Dash, dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc

from code_to_variable import scan_file, positions_file, exits_file, interface_file, app_file

###############################################################################
#
# DATA ########################################################################
#
###############################################################################

###  Global Data  *************************************************************        Global

# the files used to create the interface
df_history = pd.read_csv("history.csv")
df_positions = pd.read_csv("positions.csv")
df_all_sectors = pd.read_csv("tickers_sectors_6000.csv")

# number of current positions
num_positions = len(df_positions)

# list of stocks currently/recently owned used for candlestick and correlation graphs
first_hist = df_history.iat[0, 0]
second_hist = df_history.iat[1, 0]

if num_positions == 3:
    tickers = df_positions["STOCK"].tolist()
    displaying = "Currently Held Positions (stock tickers): "
elif num_positions == 2:
    tickers = df_positions["STOCK"].tolist()
    tickers.append(first_hist)
    displaying = "Currently/Recently Held Stocks: "
elif num_positions == 1:
    tickers = df_positions["STOCK"].tolist()
    tickers.append(first_hist)
    tickers.append(second_hist)
    displaying = "Currently/Recently Held Stocks: "
elif num_positions == 0:
    df_tickers = df_history["STOCK"].head(3)
    tickers = df_tickers["STOCK"].tolist()
    displaying = "Recently Held Positions (stock tickers): "


# function to get info from yahoo!
def ticker_histories(tickers, history):
    df = yf.download(tickers, group_by="ticker", period=history, interval="1d")
    dict = {idx: gp.xs(idx, level=0, axis=1) for idx, gp in df.groupby(level=0, axis=1)}
    return dict


# call the data function and save
trade_dict = ticker_histories(tickers, history="1y")

# get the **starting** balance from the trade history
last_row = len(df_history) - 1
opening_balance = df_history.iloc[last_row]["BALANCE"]

# and the current balance
balance = df_history.iloc[0]["BALANCE"]

# colors to be used for interface
this_green = "#198754"
this_red = "#DC3545"
this_blue = "#0dcaf0"
background = "#FCFBFC"
backgound_color = {"plot_bgcolor": "#FCFBFC"}

###  Card Data - Stats  ********************************************************        Card

# variables used for cards
pct_gain = abs((((balance - opening_balance) / opening_balance) * 100).round(2))
num_trades = df_history["STOCK"].count()
successful_trades = df_history["Profit"].where(df_history["Profit"] > 0).count()
failed_trades = num_trades - successful_trades

# calculate percent change
df_gain_loss = df_history
df_gain_loss["pct_change"] = (df_gain_loss["Income"] - df_gain_loss["Cost"]) / df_gain_loss["Cost"]

# calculate percent earned from positive valued trade incomes
gain_trades = (
        (df_gain_loss["pct_change"].where(df_gain_loss["pct_change"] > 0).mean()) * 100
).round(1)

# calculate percent loss from negative valued trade incomes
loss_trades = abs(
    (
            (df_gain_loss["pct_change"].where(df_gain_loss["pct_change"] <= 0).mean()) * 100
    ).round(1)
)

# balance card is green or red depending on current balance
if balance > opening_balance:
    border = "border-start border-success border-5"
else:
    border = "border-start border-danger border-5"

# number of trades
number_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4(f"{num_trades} Trades"),
        ],
        className="border-start border-info border-5",
    ),
    className="text-center m-2",
)

# number of successful trades with percent gain
success_card = dbc.Card(
    dbc.CardBody(
        [html.H4(f"{successful_trades} Winners"), html.H6(f"{gain_trades}% Ave Gain")],
        className="border-start border-success border-5",
    ),
    className="text-center m-2",
)

# number of unsuccessful trades with percent loss
fail_card = dbc.Card(
    dbc.CardBody(
        [html.H4(f"{failed_trades} Losers"), html.H6(f"{loss_trades}% Ave Loss")],
        className="border-start border-danger border-5",
    ),
    className="text-center m-2",
)

# current balance with percent up or down
balance_card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Balance"),
            html.H4(f"${balance}"),
            html.H6(
                f"Up {pct_gain}%" if balance > opening_balance else f"Down {pct_gain}%"
            ),
        ],
        className=border,
    ),
    className="text-center m-2",
)


###  Candlestick Data **********************************************************        Candlestick

# current day Close needs to be compared with previous day's close
# to determine color of the candlestick - regardless of current day's
# close compared with its open
def close_lag(df):
    yesterday = df["Close"].shift(1).round(3)
    df = df.assign(Yesterday=yesterday)
    return df


# this entire strategy is built on the idea that simple moving averages
# can be useful predictors of a stock's future activity
def rolling_MAs(df):
    ma_3 = df["Close"].rolling(3).mean().round(3).tolist()
    ma_6 = df["Close"].rolling(6).mean().round(3).tolist()
    ma_13 = df["Close"].rolling(13).mean().round(3).tolist()
    ma_21 = df["Close"].rolling(21).mean().round(3).tolist()
    df = df.assign(MA3=ma_3, MA6=ma_6, MA13=ma_13, MA21=ma_21)
    return df


# num is used in the naming scheme for current positions
# tickers indicate the symbols for the current positions
num = 0

# loop builds the .csv files the candlestick charts use
# when different radio buttons are selected
for trade in tickers:
    df = trade_dict[trade]
    df = close_lag(df)
    df = rolling_MAs(df)
    df = df.tail(30)
    df.to_csv(f"position_{num}.csv")
    num += 1


## The construction of the candlestick chart happens in the Callback below (beneath the Dashboard section)

###  Correlation Data  *********************************************************        Correlation

def position_correlations():
    # num is used to distinguish the first dataframe from those
    # that follow so that all the price close data can be merged
    # into one dataframe, which is the purpose of the loop
    num = 0
    for ticker in tickers:
        df = trade_dict[ticker]
        df = df[["Close"]].round(4)
        df = df.rename(columns={"Close": ticker})
        if num == 0:
            positions_close = df
        else:
            positions_close = pd.merge(
                positions_close, df, left_index=True, right_index=True
            )
        num += 1

    # create a dictionary for the three stock positions
    posi_dict = positions_close.to_dict("list")

    # get the names of the columns -- the stock symbols
    t0, t1, t2 = tickers[0], tickers[1], tickers[2]

    # determine the minimum and maximum values for each subplot column
    max_t0, max_t1, max_t2 = max(posi_dict[t0]), max(posi_dict[t1]), max(posi_dict[t2])
    min_t0, min_t1, min_t2 = min(posi_dict[t0]), min(posi_dict[t1]), min(posi_dict[t2])

    # get the correlations for each column in the positions_close dataset
    pear_corr = positions_close.corr(method="pearson")

    # the correlations for each pair of columns assigned to variables
    ticker_corr_0_1 = pear_corr[t0][t1].round(2).astype(str)
    ticker_corr_0_2 = pear_corr[t0][t2].round(2).astype(str)
    ticker_corr_1_2 = pear_corr[t1][t2].round(2).astype(str)

    # get the data for ordinary least squares lines
    # and assemble it into the lines-of-best-fit
    column_0 = list(positions_close.iloc[:, 0])
    column_1 = list(positions_close.iloc[:, 1])
    column_2 = list(positions_close.iloc[:, 2])

    array_0 = np.array(column_0).reshape(-1, 1)
    array_1 = np.array(column_1).reshape(-1, 1)
    array_2 = np.array(column_2).reshape(-1, 1)

    lm_0_1 = LinearRegression()
    lm_0_2 = LinearRegression()
    lm_1_2 = LinearRegression()

    lm_fit_0_1 = lm_0_1.fit(array_0, array_1)
    lm_fit_0_2 = lm_0_2.fit(array_0, array_2)
    lm_fit_1_2 = lm_1_2.fit(array_1, array_2)

    # intercepts and coefficients
    fit_0_1_intercept = lm_fit_0_1.intercept_
    fit_0_2_intercept = lm_fit_0_2.intercept_
    fit_1_2_intercept = lm_fit_1_2.intercept_

    fit_0_1_coef = lm_fit_0_1.coef_
    fit_0_2_coef = lm_fit_0_2.coef_
    fit_1_2_coef = lm_fit_1_2.coef_

    interc_0_1 = str(fit_0_1_intercept).lstrip('[').rstrip(']')
    interc_0_2 = str(fit_0_2_intercept).lstrip('[').rstrip(']')
    interc_1_2 = str(fit_1_2_intercept).lstrip('[').rstrip(']')

    coef_0_1 = str(fit_0_1_coef).lstrip('[').rstrip(']')
    coef_0_2 = str(fit_0_2_coef).lstrip('[').rstrip(']')
    coef_1_2 = str(fit_1_2_coef).lstrip('[').rstrip(']')

    interc_0_1 = float(interc_0_1)
    interc_0_2 = float(interc_0_2)
    interc_1_2 = float(interc_1_2)

    coef_0_1 = float(coef_0_1)
    coef_0_2 = float(coef_0_2)
    coef_1_2 = float(coef_1_2)

    # regression lines
    x0_sorted = sorted(column_0)
    x1_sorted = sorted(column_1)
    x2_sorted = sorted(column_2)

    y_regre_0_1 = [x * coef_0_1 for x in x0_sorted]
    y_regre_0_2 = [x * coef_0_2 for x in x0_sorted]
    y_regre_1_2 = [x * coef_1_2 for x in x1_sorted]

    y_regress_0_1 = [x + interc_0_1 for x in y_regre_0_1]
    y_regress_0_2 = [x + interc_0_2 for x in y_regre_0_2]
    y_regress_1_2 = [x + interc_1_2 for x in y_regre_1_2]

    ##############################################

    # the correlation matrix
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.05,
        print_grid=True,
        column_titles=tickers,
        row_titles=tickers,
        subplot_titles=(
            "",
            f"r = {ticker_corr_0_1}",
            f"r = {ticker_corr_0_2}",
            "",
            "",
            f"r = {ticker_corr_1_2}",
            "",
            "",
            "",
        ),
    )

    # plotly does not have an option for only having text in a subplot, so
    # created piece by piece -- these points are the same color as the background
    # and provide the space for the 'r = ' text that is compatible with the other
    # subplots of the matrix
    dummy_fig_0 = go.Scatter(
        x=[min_t1, min_t1, max_t1, max_t1],
        y=[min_t0, max_t0, max_t0, min_t0],
        mode="markers",
        marker_color="#FCFBFC",
    )
    dummy_fig_1 = go.Scatter(
        x=[min_t2, min_t2, max_t2, max_t2],
        y=[min_t1, max_t1, max_t1, min_t1],
        mode="markers",
        marker_color="#FCFBFC",
    )
    dummy_fig_2 = go.Scatter(
        x=[min_t2, min_t2, max_t2, max_t2],
        y=[min_t2, max_t2, max_t2, min_t2],
        mode="markers",
        marker_color="#FCFBFC",
    )

    # plotly code for the LOWER TRIANGLE of scatter plots
    scat_0 = go.Scatter(
        x=column_0, y=column_1,
        mode='markers', marker_color=this_blue
    )

    scat_1 = go.Scatter(
        x=column_0, y=column_2,
        mode='markers', marker_color=this_blue
    )

    scat_2 = go.Scatter(
        x=column_1, y=column_2,
        mode='markers', marker_color=this_blue
    )

    # plotly code for the regression lines in the scatter plots
    # the regression lines are affected by data in other subplots
    # and remains an unresolved issue
    plot_ols_0 = go.Scatter(
        x=x0_sorted, y=y_regress_0_1,
        mode='lines', marker_color="black",
        line={'width': 1.5}
    )

    plot_ols_1 = go.Scatter(
        x=x0_sorted, y=y_regress_0_2,
        mode='lines', marker_color="black",
        line={'width': 1.5}
    )

    plot_ols_2 = go.Scatter(
        x=x1_sorted, y=y_regress_1_2,
        mode='lines', marker_color="black",
        line={'width': 1.5}
    )

    # plotly code for the DIAGONAL of price histograms
    hist_0 = go.Histogram(
        x=posi_dict[t0],
        y=posi_dict[t0],
        marker_color=this_blue,
        marker_line_width=1,
        marker_line_color="black",
    )

    hist_1 = go.Histogram(
        x=posi_dict[t1],
        y=posi_dict[t1],
        marker_color=this_blue,
        marker_line_width=1,
        marker_line_color="black",
    )

    hist_2 = go.Histogram(
        x=posi_dict[t2],
        y=posi_dict[t2],
        marker_color=this_blue,
        marker_line_width=1,
        marker_line_color="black",
    )

    # adding the individual graphs to the correlation matrix
    fig.add_trace(hist_0, row=1, col=1)
    fig.add_trace(hist_1, row=2, col=2)
    fig.add_trace(hist_2, row=3, col=3)

    fig.add_trace(scat_0, row=2, col=1)
    fig.add_trace(scat_1, row=3, col=1)
    fig.add_trace(scat_2, row=3, col=2)
    fig.add_trace(plot_ols_0, row=2, col=1)
    fig.add_trace(plot_ols_1, row=3, col=1)
    fig.add_trace(plot_ols_2, row=3, col=2)

    fig.add_trace(dummy_fig_0, row=1, col=2)
    fig.add_trace(dummy_fig_1, row=1, col=3)
    fig.add_trace(dummy_fig_2, row=2, col=3)

    # 'r = ' has to be placed on the matrix grid by percent
    fig.update_annotations(y=0.825, selector={"text": f"r = {ticker_corr_0_1}"})
    fig.update_annotations(y=0.825, selector={"text": f"r = {ticker_corr_0_2}"})
    fig.update_annotations(y=0.480, selector={"text": f"r = {ticker_corr_1_2}"})

    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis2_visible=False, yaxis3_visible=False, yaxis6_visible=False)
    fig.update_layout(
        margin=dict(l=15, r=15, t=30, b=0),
    )
    fig.update_layout(backgound_color)

    corr_matrix = fig

    return corr_matrix


###  Line Chart Data  **********************************************************        Line

def vs_indices():
    # vs_indices creates a line chart for the three major indices
    # and this program's trading history as percent change since
    # the trading history's beginning

    # get the dates from the trade history to use with indices
    df_h = pd.read_csv("history.csv")
    df_h.drop_duplicates(subset=['Sell Date'], keep='first', inplace=True)

    here = len(df_h) - 1
    start = df_h.iloc[here]['Buy Date']
    end = df_h.iloc[0]['Sell Date']
    end = pd.to_datetime(end, format='%Y-%m-%d')
    one_day = pd.Timedelta(1, "D")
    end = str(end + one_day)
    end = pd.to_datetime(end)

    # tickers for S&P 500, Dow Jones, and the Nasdaq
    indices = ["^GSPC", "^DJI", "^IXIC"]

    # price information for indices / formatting for merge
    df = yf.download(indices, group_by="Ticker", start=start, end=end)
    df = df.stack(level=0).rename_axis(["Sell Date", "STOCK"]).reset_index(level=1)
    not_needed = ["High", "Low", "Adj Close", "Open", "Volume"]
    df = df.drop(not_needed, axis=1)
    df.reset_index(inplace=True)

    df = df.pivot(index='Sell Date', columns='STOCK', values='Close')
    df = df.reset_index()
    df_h['Sell Date'] = pd.to_datetime(df_h['Sell Date'].astype(str), format='%Y-%m-%d')

    # merge daily Close from indices with Trade History
    df_compare = pd.merge(df, df_h, on="Sell Date")
    df_compare = df_compare.reindex(columns=["Sell Date", "^GSPC", "^DJI", "^IXIC", "BALANCE"])

    # rename columns for convenience
    df_compare = df_compare.rename(
        columns={
            "Sell Date": "Date",
            "^GSPC": "S&P500",
            "^DJI": "Dow",
            "^IXIC": "Nasdaq",
            "BALANCE": "Trades",
        }
    )

    # assign each starting value to a variables
    # then calculate each day's percent change
    # from that start date
    df = df_compare.set_index("Date")
    init_snp = df.iloc[0]["S&P500"]
    init_dow = df.iloc[0]["Dow"]
    init_nasdaq = df.iloc[0]["Nasdaq"]
    init_trades = df.iloc[0]["Trades"]
    df["S&P500_pct"] = (df["S&P500"] - init_snp) / init_snp
    df["Dow_pct"] = (df["Dow"] - init_dow) / init_dow
    df["Nasdaq_pct"] = (df["Nasdaq"] - init_nasdaq) / init_nasdaq
    df["Trades_pct"] = (df["Trades"] - init_trades) / init_trades

    # plotly code for each line from data created above
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Dow_pct,
        mode='lines',
        name='Dow',
        line=dict(color=this_green,
                  width=1,
                  dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['S&P500_pct'],
        mode='lines',
        name='S&P 500',
        line=dict(color=this_red,
                  width=1,
                  dash='dot'
                  )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Nasdaq_pct,
        mode='lines',
        name='Nasdaq',
        line=dict(color=this_blue,
                  width=1,
                  dash='dashdot')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Trades_pct,
        mode='lines',
        name='Trades',
        line=dict(color='black',
                  width=1)
    ))

    fig.update_layout(
        backgound_color,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.25
        ),
        legend=dict(yanchor="top",
                    y=1.07,
                    xanchor="left",
                    x=0.0,
                    font=dict(size=11),
                    orientation="h",
                    groupclick="toggleitem")
    )

    return fig


###  Histogram Data  ***********************************************************        Histogram

def price_freq():
    # does this strategy prefer a certain price range?

    # a simple histogram of closing price data
    histo = go.Figure(
        data=[
            go.Histogram(
                x=df_history["Share Price"],
                marker_color=this_green,
                marker_line_width=1,
                marker_line_color="black",
            )
        ]
    )
    histo.update_layout(
        backgound_color,
        margin=dict(l=20, r=0, t=5, b=50),
        xaxis_title="Dollars",
        yaxis_title="Number of Trades",
    )

    return histo


###  Treemap Data  *************************************************************        Treemap

def sector_tree():
    # backtesting reveals that some sectors conform to this strategy
    # better than others

    # the treemap takes a tally of traded sectors and displays it as a percentage
    history_sector = df_history
    history_sector = history_sector[["STOCK", "Sector"]]
    history_sector = history_sector.groupby("Sector").count().reset_index(level=0)
    history_sector = history_sector.rename(
        columns={"SECTOR": "Sector", "STOCK": "Count"}
    )
    history_sector["pct"] = (
            (history_sector["Count"] / history_sector.Count.sum()) * 100
    ).round(2)

    # convert two relevant columns of history_sector into dictionary
    hist_sect = history_sector[["Sector", "pct"]]
    sect_dict = hist_sect.to_dict("list")

    # plotly code for putting together the treemap data
    # the treemap is configured from lists with corresponding elements
    parents, labels, values, color = (
        [""],
        ["Percentages of Traded Sectors"],
        [0],
        ["black"],
    )

    # the outermost section of the treemap contains the inner sections called labels
    # since there is only one outer section, each label belongs to the same parents
    # so that parent must be named once for each label in the parent list, hence the for loop
    for i in range(len(hist_sect)):
        parents.append("Percentages of Traded Sectors")

    # text color can be assigned to each label with a list of corresponding hex codes
    # the outermost section is black (and title-like) while the remainder are white
    for i in range(len(hist_sect)):
        color.append("#FFFFFF")

    # the key_list is a list of all the traded sectors
    key_list = [k for k in sect_dict["Sector"]]

    # the value_list is a list of all the percentages for the sectors
    value_list = [v for v in sect_dict["pct"]]

    # the keys and values are added to their appropriate list
    labels.extend(key_list)
    values.extend(value_list)

    # plotly code for the figure
    tree = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+percent root",
            textposition="middle center",
            textfont={"size": 16, "color": color},
            root_color=background,
        )
    )

    tree.update_layout(
        treemapcolorway=[this_blue, this_red, this_green],
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return tree


###  Positions / Trade History Data Tables  ************************************        Data Tables

# use the csv files for plotly data tables

# minor changes to dataframe for interface
current_positions = df_positions
current_positions = current_positions.drop(columns=["Hit_94%", "Hit_97.5%", 'High',
                                                    'Sell On', 'Pattern', 'Activity',
                                                    'Strength','Multiplier', 'P'])
current_positions["Target Price"] = current_positions["Target Price"].round(2)

if len(current_positions) == 0:
    diff_dataset = "There are better days for trades. Sitting this one out until the market shapes up. \
        No new positions."
    padding = 30
else:
    diff_dataset = ""
    padding = 0

# issue with df_history picking up the pct_change feature ??
# may be VS Code issue
df_exit_history = pd.read_csv("history.csv")
df_exit_history["Percent Gain"] = (
        (df_exit_history["Income"] - df_exit_history["Cost"]) / df_exit_history["Cost"]
).apply('{:.1%}'.format)
df_exit_history = df_exit_history[
    [
        "STOCK",
        "Sector",
        "Buy Date",
        "Sell Date",
        "Shares",
        "Share Price",
        "Sell Price",
        "Cost",
        "Income",
        "Profit",
        "Percent Gain",
        "BALANCE",
    ]
]

###############################################################################
#
# DASHBOARD ###################################################################
#
###############################################################################

# dash and dash_bootstrap_components are used to arrange the information for the
# interface in rows, then columns, and in one case, rows again

stylesheet = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__,
    external_stylesheets=stylesheet,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

server = app.server

# the main container for the dashboard
app.layout = dbc.Container(
    [
        ### FIRST ROW
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        "Swing Trades by Simple Moving Averages:",
                        className="text-center",
                    ),
                )
            ]
        ),
        ### SECOND ROW
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        "A pandas, Plotly, and Dash Project", className="text-center"
                    )
                )
            ]
        ),
        ### THIRD ROW
        dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        "Depending on the market, between zero and three stocks are traded on a daily basis. \
                        Buy signals are generated when both of the following two conditions are met: 1) \
                        the closing price is greater than \
                        the 3-day moving average, and 2) the 3-day moving average is greater than \
                        the 6-, 13-, and 21-day moving averages. \
                        Trades are determined by the stock's likelihood to reach a\
                        certain price point in the next trading session. Exit levels \
                        are generated based on daily highs following buy signals from the \
                        stock's trading history. All positions sold within two day."
                    ),
                    style={"padding-bottom": 20},
                ),
            ]
        ),
        ### FOURTH ROW
        dbc.Row(
            [
                ## R4 - First Column - basic stats in Cards
                dbc.Col(
                    [
                        dbc.Col(number_card),
                        dbc.Col(success_card),
                        dbc.Col(fail_card),
                        dbc.Col(balance_card),
                    ],
                    width=2,
                ),
                ## R4 - Second Column - two figures
                dbc.Col(
                    [
                        # Current Positions Data Table
                        dbc.Row(
                            [
                                html.H4("Current Positions"),
                                dash_table.DataTable(
                                    current_positions.to_dict("records"),
                                    [
                                        {"name": i, "id": i}
                                        for i in current_positions.columns
                                    ],
                                    fixed_rows={"headers": True},
                                    style_table={
                                        "margin": "auto",
                                        "justify-content": "space-around",
                                        "overflowX": "auto",
                                        "maxHeight": "150px",
                                        "border-style": "hidden",
                                        "box-shadow": "0 0 0 1px black",
                                        "margin-bottom": "30px",
                                    },
                                    style_cell={
                                        "height": "auto",
                                        "whiteSpace": "normal",
                                    },
                                    style_cell_conditional=[
                                        {'if': {'column_id': 'STOCK'},
                                         'width': '60px'},
                                        {'if': {'column_id': 'Buy Date'},
                                         'width': '100px'},
                                        {'if': {'column_id': 'Shares'},
                                         'width': '60px'},
                                        {'if': {'column_id': 'Target Price'},
                                         'width': '70px'},
                                    ],
                                    fill_width=True,
                                ),
                                html.H6(diff_dataset, style={"padding-bottom": padding}, )
                            ]
                        ),
                        # Price Histogram
                        dbc.Row(
                            [
                                html.H6(
                                    "What price range is conducive to this approach?"
                                ),
                                dcc.Graph(
                                    id="histogram",
                                    figure=price_freq(),
                                    config={"displayModeBar": False},
                                ),
                            ],
                            style={"height": "58%"},
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                ),
                ## R4 - Third Column
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col([html.H5(displaying,
                                             style={"width": "345px",
                                                    # "padding-left": "15px",
                                                    })]),
                            dbc.Col([
                                # radio buttons for selecting which stock to view in candlestick chart
                                dcc.RadioItems(
                                    options=[
                                        {"label": tickers[0], "value": "position_0"},
                                        {"label": tickers[1], "value": "position_1"},
                                        {"label": tickers[2], "value": "position_2"},
                                    ],
                                    id="position",
                                    value="position_0",
                                    inline=True,
                                    style={
                                        "display": "flex",
                                        "margin": "auto",
                                        "width": "265px",
                                        "justify-content": "space-around",
                                        "padding-right": "45px",
                                    },
                                ),
                            ]),

                        ]),

                        dbc.Row([  # candlesticks with volume data per current positions
                            dcc.Graph(
                                id="candles",
                                figure={},
                                style={"paper_bgcolor": "#ECF3F9"},
                                config={"displayModeBar": False},
                            ),
                        ]),

                    ],
                    style={"width": "50%"},
                    xs=12,
                    sm=12,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                ),
            ],
            style={"padding-bottom": 25},
        ),
        ### FIFTH ROW
        dbc.Row(
            [
                ## R5 - First Column - percent change from start date to current for indices and trades
                dbc.Col(
                    [
                        html.H5("What's the relationship with the major indices?"),
                        dcc.Graph(
                            id="comparisons",
                            figure=vs_indices(),
                            config={"displayModeBar": False},
                            style={"padding-top": 17, 'height': '75%'}
                        ),
                        html.P(
                            "The Trades show a hypersensitivity to market fluctuations. \
                            More data may smooth out a bit of the wild swings, \
                            but their bearing relative to the markets may still change sporadically."
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                ),
                ## R5 - Second Column - treemap of sectors traded
                dbc.Col(
                    [
                        dcc.Graph(
                            id="sectors",
                            figure=sector_tree(),
                            config={"displayModeBar": False},
                            style={"height": "85%"},
                        ),
                        html.P(
                            "The treemap illustrates how often individual sectors have been traded. \
                            The markets comprise a total of 11 sectors."
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                ),
                ## R5 - Third Column - correlation matrix
                dbc.Col(
                    [
                        html.H5("Correlation Matrix for Current Positions"),
                        dcc.Graph(
                            id="corr_matrix",
                            figure=position_correlations(),
                            style={"paper_bgcolor": "#ECF3F9"},
                            config={"displayModeBar": False},
                        ),
                        html.P(
                            "The histograms shed some light on the stock's price history. \
                       The scatter plots indicate how closely each pair are correlated. \
                       Ideally, r-values are closer to zero than 1 or -1 so that trading \
                       risk is spread out among multiple stocks."
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                ),
            ],
            style={"padding-top": 17, "padding-bottom": 10},
        ),
        ### SIXTH ROW - trade history
        dbc.Row(
            [
                html.H4("Trade History"),
                dash_table.DataTable(
                    df_exit_history.to_dict("records"),
                    [{"name": i, "id": i} for i in df_exit_history.columns],
                    fixed_rows={"headers": True},
                    style_table={
                        "margin": "auto",
                        "justify-content": "space-around",
                        "overflowX": "auto",
                        "maxHeight": "290px",
                        "border-style": "hidden",
                        "box-shadow": "0 0 0 1px black",
                        "margin-bottom": "30px",
                    },
                    style_cell={"height": "auto", "whiteSpace": "normal"},
                    style_cell_conditional=[
                        {"if": {"column_id": "STOCK"}, "width": "60px"},
                        {"if": {"column_id": "Sector"}, "width": "200px"},
                        {"if": {"column_id": "Shares"}, "width": "60px"},
                        {"if": {"column_id": "Percent Gain"}, "width": "70px"},
                    ],
                    fill_width=True,
                ),
            ],
            style={"paper_bgcolor": "#ECF3F9"},
        ),
        ### SEVENTH ROW - update information and where to view this code
        dbc.Row(
            [
                html.H6(
                    "This dashboard updates in the evening after each trading session.",
                    className="text-center",
                ),

                html.A(
                    "Find me on LinkedIn.",
                    href="https://www.linkedin.com/in/phil-norris-data-science/",
                    className="text-center",
                    target="_blank"
                ),
            ],
            style={"padding-bottom": 75},
        ),

        ### EIGHTH ROW - tabs with the code for data and interface
        dbc.Row([
            dbc.Col([
                html.H6(
                    "The code that generates the data and runs this dashboard can be viewed below."
                ),
                dcc.Tabs([
                    dcc.Tab(label='scan.py', children=[
                        scan_file
                    ]),
                    dcc.Tab(label='positions.py', children=[
                        positions_file
                    ]),
                    dcc.Tab(label='exits.py', children=[
                        exits_file
                    ]),
                    dcc.Tab(label='interface.py', children=[
                        interface_file
                    ]),
                    dcc.Tab(label='app.py', children=[
                        app_file
                    ]),
                ]),

            ],
                xs=12,
                sm=12,
                md=8,
                lg=8,
                xl=8,
                xxl=8,
            ),
        ]),
    ]
)


###############################################################################
#
# CALLBACK ####################################################################
#
###############################################################################

### CANDLESTICKS - see DATA section above to candlestick data
@callback(
    Output("candles", "figure"),
    Input("position", "value")  # from below to the dashboard
)  # from radio buttons to figure below
def plot_candles(position):
    # retrieve selected stock data
    this_data = f"{position}.csv"
    df = pd.read_csv(this_data)
    df["Date"] = pd.to_datetime(df["Date"])

    # there are five types of candlesticks: 2 green, 2 red, 1 black

    # green - better than yesterday
    inc_up = df[(df["Close"] > df["Open"]) & (df["Close"] > df["Yesterday"])]
    inc_down = df[(df["Close"] < df["Open"]) & (df["Close"] > df["Yesterday"])]

    # red - not as good as yesterday
    dec_up = df[(df["Close"] < df["Open"]) & (df["Close"] < df["Yesterday"])]
    dec_down = df[(df["Close"] > df["Open"]) & (df["Close"] < df["Yesterday"])]

    # black - same as yesterday
    flat = df[df["Close"] == df["Open"]]

    ## plotly code for the candlesticks

    # close is greater than open and previous close
    green_up = go.Candlestick(
        x=inc_up["Date"],
        open=inc_up["Open"],
        high=inc_up["High"],
        low=inc_up["Low"],
        close=inc_up["Close"],
        increasing_line_color=this_green,
        increasing_fillcolor="white",
        decreasing_line_color=this_green,
        decreasing_fillcolor="white",
        name="Gain/Up",
        showlegend=False,
    )

    # close is less than open but greater than previous close
    green_down = go.Candlestick(
        x=inc_down["Date"],
        open=inc_down["Open"],
        high=inc_down["High"],
        low=inc_down["Low"],
        close=inc_down["Close"],
        increasing_line_color=this_green,
        increasing_fillcolor=this_green,
        decreasing_line_color=this_green,
        decreasing_fillcolor=this_green,
        name="Gain/Down",
        showlegend=False,
    )

    # close is less than open and previous close
    red_up = go.Candlestick(
        x=dec_up["Date"],
        open=dec_up["Open"],
        high=dec_up["High"],
        low=dec_up["Low"],
        close=dec_up["Close"],
        increasing_line_color=this_red,
        increasing_fillcolor=this_red,
        decreasing_line_color=this_red,
        decreasing_fillcolor=this_red,
        name="Loss/Up",
        showlegend=False,
    )

    # close is greater than open but less than previous close
    red_down = go.Candlestick(
        x=dec_down["Date"],
        open=dec_down["Open"],
        high=dec_down["High"],
        low=dec_down["Low"],
        close=dec_down["Close"],
        increasing_line_color=this_red,
        increasing_fillcolor="white",
        decreasing_line_color=this_red,
        decreasing_fillcolor="white",
        name="Loss/Down",
        showlegend=False,
    )

    # close is the same as the open
    flat = go.Candlestick(
        x=flat["Date"],
        open=flat["Open"],
        high=flat["High"],
        low=flat["Low"],
        close=flat["Close"],
        increasing_line_color="black",
        increasing_fillcolor="black",
        decreasing_line_color="black",
        decreasing_fillcolor="black",
        name="Flat",
        showlegend=False,
    )

    # establish a figure with 3 parts candlestick, 1 part volume
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        specs=[[{"rowspan": 3}], [None], [None], [{}]],
        print_grid=True,
    )

    # plotly data from Data section above is appended to the figure
    fig.add_trace(green_up, row=1, col=1)
    fig.add_trace(green_down, row=1, col=1)
    fig.add_trace(red_up, row=1, col=1)
    fig.add_trace(red_down, row=1, col=1)
    fig.add_trace(flat, row=1, col=1)

    # 3-day moving average line - on a toggle
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA3"],
            mode="lines",
            name="3-Day",
            line={"width": 1, "dash": "solid", "color": "black"},
        ),
        row=1,
        col=1,
    )

    # 6-day moving average line - on a toggle
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA6"],
            mode="lines",
            name="6-Day",
            line={"width": 1, "dash": "dash", "color": this_blue},
        ),
        row=1,
        col=1,
    )

    # 13-day moving average line - on a toggle
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA13"],
            mode="lines",
            name="13-Day",
            line={"width": 1, "dash": "dash", "color": this_blue},
        ),
        row=1,
        col=1,
    )

    # 21-day moving average line - on a toggle
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA21"],
            mode="lines",
            name="21-Day",
            line={"width": 1, "dash": "dash", "color": this_blue},
        ),
        row=1,
        col=1,
    )

    # volume bars
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color=this_blue,
            showlegend=False,
            marker_line_color="black",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(legend_title_text="Moving Averages:")
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.07,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            orientation="h",
            groupclick="toggleitem",
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(backgound_color)
    fig.update_layout(backgound_color, yaxis_title="Dollars")

    return fig


if __name__ == "__main__":
    app.run_server(debug=False)

''',
                           language="python",
                           withLineNumbers=True,
                           )

app_file = dmc.Prism('''
"""
    This files automates the execution of the
    exits.py, scan.py, and positions.py files
    on days when the markets are open, at the
    appropriate times.
"""

import pandas as pd
import pandas_market_calendars as mc
import schedule
import subprocess
import time

print("\\n", "Starting the Schedule", "\\n")


# update stock tickers on Mondays for the week
def weekly_tickers():
    subprocess.run(["python", "tickers.py"])
    return schedule.CancelJob


# main function from which other functions run
# when time and date satisfy the 'schedule' mechanism,
# this function is called to execute stock trading files
def market_hours():
    # execute the trading file
    def selling():
        print('Selling')
        subprocess.run(["python", "exits.py"])
        return schedule.CancelJob

    def scaning():
        print('Scaning')
        subprocess.run(["python", "scan.py"])
        return schedule.CancelJob

    def buying():
        print('Buying')
        subprocess.run(["python", "positions.py"])
        return schedule.CancelJob

    # start time is current time from which end time is determined
    start_date = pd.Timestamp.now()
    seven_days = pd.Timedelta(7, "D")
    end_date = start_date + seven_days

    # to avoid running the python files every day
    # and creating duplicate information, the
    # schedule will follow the NYSE's schedule
    schd = mc.get_calendar("NYSE")

    # create dataframe with start_date and end_date from above
    open_time = schd.schedule(start_date=start_date, end_date=end_date)

    # add columns to df for time in hours, minutes, and seconds
    time_zone_correction = pd.Timedelta(4, "hours")  # adjust by 1 hour for DST
    one_minute = pd.Timedelta(1, "minutes")
    two_minutes = pd.Timedelta(2, "minutes")
    three_minutes = pd.Timedelta(3, "minutes")
    forty_minutes = pd.Timedelta(40, "minutes")
    twelve_seconds = pd.Timedelta(12, "seconds")

    open_time["market_open"] = open_time["market_open"] - time_zone_correction
    open_time["market_close"] = open_time["market_close"] - time_zone_correction
    open_time["at_09_30"] = (open_time["market_open"]).dt.strftime("%H:%M")
    open_time["at_10_10"] = (open_time["market_open"] + forty_minutes).dt.strftime("%H:%M")
    open_time["at_three_til"] = (open_time["market_close"] - three_minutes).dt.strftime("%H:%M")
    open_time["at_two_til"] = (open_time["market_close"] - two_minutes).dt.strftime("%H:%M")
    open_time["at_one_til"] = (open_time["market_close"] - one_minute).dt.strftime("%H:%M")
    open_time["buy_now"] = (open_time["market_close"] - twelve_seconds).dt.strftime('%H:%M:%S')
    open_time["date"] = (open_time["market_open"]).dt.strftime("%Y-%m-%d")

    open_time = open_time[
        [
            "date",
            "at_09_30",
            "at_10_10",
            "at_three_til",
            "at_two_til",
            "at_one_til",
            "buy_now",
            "market_open",
            "market_close",
        ]
    ]
    open_time = open_time.reset_index(drop=True)

    print("\\nIt is now", start_date, "and market hours for the next few days are\\n", 
        open_time,"\\n")
    open_time.to_csv("src/hours.csv", index=False)

    # save times as variables
    selling_time = open_time.loc[0, "at_09_30"]
    scan_time = open_time.loc[0, "at_one_til"]
    buying_time = open_time.loc[0, "buy_now"]

    date_today = str(start_date.date())
    first_market_date = open_time.iloc[0, 0]

    trading_day = True if (date_today == first_market_date) else False

    # selling_time = '17:58:00'

    if trading_day:
        print('\\nTrading Day Begins')
        schedule.every().day.at(selling_time).do(selling)
        schedule.every().day.at(scan_time).do(scaning)
        schedule.every().day.at(buying_time).do(buying)


# master schedule that starts the process
schedule.every().monday.at("08:45").do(weekly_tickers)
schedule.every().monday.at("09:29:45").do(market_hours)
schedule.every().tuesday.at("09:29:45").do(market_hours)
schedule.every().wednesday.at("09:29:45").do(market_hours)
schedule.every().thursday.at("09:29:45").do(market_hours)
schedule.every().friday.at("09:29:45").do(market_hours)

while True:
    schedule.run_pending()
    time.sleep(1)


''',
                     language="python",
                     withLineNumbers=True,
                     )
