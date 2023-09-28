import pandas as pd
import numpy as np
import yfinance as yf
import statistics

# terminal notes establishing the run execution
print("\n", "Today's Stocks")
time1 = pd.Timestamp.now()
time1 = time1.floor(freq="S")
print("\n", time1)

# create the dataframe where all relevant information
# will ultimately be stored for other files to use
final = pd.DataFrame()

# net is used to test a desired exit price for trades
# but is not used to calculate the actual exit price
# Close Price * net = Exit Price
net = 1.009


# function to get stock data from yfinance
def ticker_histories(tickers, history):
    df = yf.download(tickers, group_by="ticker", period=history)
    dict = {idx: gp.xs(idx, level=0, axis=1) for idx, gp in df.groupby(level=0, axis=1)}
    return dict


# does the stock meet the minimum price requirement?
def price_threshold(df):
    ave_price_1 = df['Close'].rolling(15).mean()
    ave_price_2 = df['Close'].rolling(100).mean()
    df = df.assign(ave_price_1=ave_price_1)
    df = df.assign(ave_price_2=ave_price_2)
    p_1 = df['ave_price_1'].to_numpy()
    p_2 = df['ave_price_2'].to_numpy()

    ap_1 = p_1[-1]
    ap_2 = p_2[-1]

    if (ap_1 >= 1) & (ap_2 >= 1):
        return True
    return False


# does the stock meet the minimum volume requirement?
def volume_threshold(df):
    Vol40 = df["Volume"].rolling(40).mean()
    df = df.assign(Vol40=Vol40)
    V = df["Vol40"].to_numpy()
    AV = V[-1]
    if AV >= 350_000:
        return True
    return False


# does the stock meet the minimum volatility requirement?
def volatility_threshold(df):
    volati_list = df['Close'].tolist()
    try:
        volati_year = statistics.stdev(volati_list)
    except:
        return False
    volati_month = statistics.stdev(volati_list[-20:])
    if volati_month < volati_year * 1.5:
        return True
    return False


# is the stock experiencing outlier price levels?
def outlier_threshold(df):
    close_list = df["Close"].tolist()
    outlier_list = np.array(close_list)
    mean = np.mean(outlier_list, axis=0)
    sd = np.std(outlier_list, axis=0)
    clean_list = [x for x in close_list if (x > mean - 2 * sd)]
    clean_list = [x for x in clean_list if (x < mean + 2 * sd)]
    clean_mean = np.mean(clean_list, axis=0)

    month_list = df['Close'].tolist()
    month_list = month_list[-20:]
    month_mean = np.mean(month_list, axis=0)

    if (month_mean < clean_mean * 1.75) & (month_mean > clean_mean * 0.5):
        return True
    return False


# assign the current stock's symbol to the new row list, nuro
def what_stock(ticker):
    nuro = []
    nuro.append(ticker)
    return nuro


# add the moving averages to the dataframe
# that will be used in signal generation
def rolling_MAs(df):
    ma_3 = df["Close"].rolling(3).mean().round(3)
    ma_6 = df["Close"].rolling(6).mean().round(3)
    ma_13 = df["Close"].rolling(13).mean().round(3)
    ma_21 = df["Close"].rolling(21).mean().round(3)
    df = df.assign(MA3=ma_3, MA6=ma_6, MA13=ma_13, MA21=ma_21)
    return df


# indicates in the stock's history where the 3-day moving average
# exceeds the 6-, 13-, and 21-day moving averages AND label those
# instances as a signal -- Gain or Loss
def the_test(df):
    # is the next day gain at least "net" of a percent?
    df["Target"] = (df["Close"]) * net
    df["Max1"] = df["High"].shift(-1).round(2)
    df["Max2"] = df["High"].shift(-2).round(2)  # needed in take_the_hit()
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


# how well does the_test() work?
# determines the probability of successfully generated signals
def p(df, nuro):
    px = df["Trigger"].to_numpy()  # make a list of Gains/Losses
    px = px[:-1]  # drop that last element (needs tomorrow's data)
    G009 = np.count_nonzero(px == "Gain")  # count gains
    L009 = np.count_nonzero(px == "Loss")  # count losses
    if G009 | L009 == 0:  # go to next loop iteration is either is zero
        return -1, -1, -1
    p_tally = G009 / (G009 + L009)  # find probability of gains
    if p_tally < 0.8:  # reject probabilities below 80%
        return -1, -1, -1
    nuro.append(p_tally)  # add value to the new row
    return p_tally, px, nuro


# buy signal will only generate according to the follow
# scenarios of 0s, Gains, and Loss
def the_signal(Px009, nuro):
    four_from_last = Px009[-4]
    antepenultimate = Px009[-3]
    penultimate = Px009[-2]
    ultimate = Px009[-1]
    if (
            (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "0")
            & (ultimate == "Loss")
    ):
        buy = "Yes"
    elif (
            (four_from_last == "0")
            & (antepenultimate == "0")
            & (penultimate == "Gain")
            & (ultimate == "Loss")
    ):
        buy = "Yes"
    elif (
            (four_from_last == "0")
            & (antepenultimate == "Gain")
            & (penultimate == "Gain")
            & (ultimate == "Loss")
    ):
        buy = "Yes"
    else:
        buy = "No"
    sig = [four_from_last, antepenultimate, penultimate, ultimate]
    nuro.append(sig)
    nuro.append(buy)
    return nuro


# how often does this stock generate a buy signal that
# results in a Gain? -- the more the better
# there is a 20% threshold built into this function
def occurrences(px, nuro):
    zeros = np.count_nonzero(px == "0")
    array_length = len(px)
    activity = 1 - (zeros / (array_length))
    if activity < 0.1999:
        return -1, -1
    nuro.append(activity)
    return activity, nuro


# scores the p-value with how actively it generates signals
# times 200 for roughly a 100-point scale
def strength(activity, p_tally, nuro):
    strength = round(((p_tally * activity) * 200), 1)
    nuro.append(strength)
    return nuro


# save these values for other files to use
# especially helpful in the terminal
def high_close(df, nuro):
    h = df["High"].astype(float)
    high = h.iloc[-1]
    c = df["Close"].astype(float)
    close = c.iloc[-1]
    nuro.append(high)
    nuro.append(close)
    return nuro


# determine the sell price based on gain history
def safe_gain(df, px, nuro):
    # create a list for percent high above close
    target = []
    pxClose = df["Close"].astype(float)
    pxHigh = df["High"].astype(float)

    # get the next high above the current close
    # and add to the target list
    for i in range(len(px) - 1):
        if px[i] == "Gain":
            x = (pxHigh.iloc[i + 1] - pxClose.iloc[i]) / pxHigh.iloc[i + 1]
            target.append(x)

    # the sell price is determined by finding an amount that's
    # greater than the close - a portion of the target list's range
    # provides a historically safe exit point
    if len(target) > 5:
        sell_inc = target[0] + ((target[-1] - target[0]) * .275)  # percentage of the range
        sell_price = pxClose.iloc[-1] * (1 + sell_inc)
    else:
        sell_price = pxClose.iloc[-1] * (net - 0.002)  # in case test period changes
    nuro.append(sell_price)
    return pxClose, nuro


# determine the exit price based on loss history
def take_the_hit(pxClose, nuro, df):
    # loss on first day after close
    df["loss_pct_1"] = np.where(
        (df["Trigger"] == "Loss"), ((df["Max1"] - df["Close"]) / df["Close"]), 0
    )

    # loss on second day after close
    df["loss_pct_2"] = np.where(
        (df["Trigger"] == "Loss"), ((df["Max2"] - df["Close"]) / df["Close"]), 0
    )

    # occasionally, a stock will generate no losses
    # so, to avoid dividing by zero:
    l0 = list(filter(None, df["loss_pct_1"]))
    l1 = list(filter(None, df["loss_pct_2"]))
    l1 = [n for n in l1 if n < 0]
    if l0 == 0:
        return -1, -1, -1
    if l1 == 0:
        return -1, -1, -1

    # get the mean for each day after close
    loss_mean_1 = sum(l0) / len(l0)
    loss_mean_2 = sum(l1) / len(l0)
    hit_pct = ((loss_mean_1 + loss_mean_2) / 2) * 0.25  # percentage of the mean

    # safe-guards
    if hit_pct < -0.0225:
        hit_pct = -0.0225
    if np.isnan(hit_pct):
        hit_pct = -0.0225  # in case of yfinance error

    take_hit = pxClose.iloc[-1] * (1 + hit_pct)  # this is the losing exit price target
    nuro.append(take_hit)
    return nuro


# this feature is for observational purposes only
def sale_target(pxClose, sell_price, nuro):
    last_close = pxClose[-1]
    sell_pct = (sell_price - last_close) / last_close
    nuro.append(sell_pct)
    return nuro


# OPERATION ***********************************************************

Tickers = pd.read_csv("tickers_sectors_6000.csv")

tickers = Tickers["SYMBOL"].to_list()
dicti_stocks = ticker_histories(tickers, history="282d")

for ticker in tickers:
    df = dicti_stocks[ticker]

    # filter
    x1 = price_threshold(df)
    if not x1:
        continue  # to avoid junky stocks

    # filter
    x2 = volume_threshold(df)
    if not x2:
        continue  # to avoid stocks that do not trade
        # often enough for quick exits

    # filter
    x3 = volatility_threshold(df)
    if not x3:
        continue  # to avoid unnecessary risk

    # filter
    x4 = outlier_threshold(df)
    if not x4:
        continue  # to avoid unnecessary risk

    # STOCK
    nuro = what_stock(ticker)

    # Moving Averages
    df = rolling_MAs(df)

    # assign Gain or Loss based on Moving Averages
    df = the_test(df)

    # probability of a Gain not being a Loss
    p_tally, px, nuro = p(df, nuro)
    if p_tally == -1:
        continue  # to avoid dividing by zero

    # Buy - when to produce a signal and to not
    # make a trade at the end of its signal streak
    nuro = the_signal(px, nuro)

    # Activity - how often the 3-day MA crosses all the others
    activity, nuro = occurrences(px, nuro)
    if activity == -1:
        continue  # to avoid stocks that do not generate enough data

    # Strength - metric devised to indicate the best candidates for trades
    nuro = strength(activity, p_tally, nuro)

    # adding HIGH and CLOSE to the final record
    nuro = high_close(df, nuro)

    # SellAt - upside price for exiting the position
    # based on sufficient history or .2% less than the test
    pxClose, nuro = safe_gain(df, px, nuro)

    # HitAt - downside price for exiting the position
    # based on sufficient history and a max of 2.25% loss
    nuro = take_the_hit(pxClose, nuro, df)

    temp_df = pd.DataFrame(
        [nuro],
        columns=[
            "STOCK",
            "P",
            "sig",
            "BUY",
            "Activity",
            "Strength",
            "HIGH",
            "CLOSE",
            "SellAt",
            "HitAt",
        ],
    )
    # add the new row (nuro) to the final dataframe by concatenation
    final = pd.concat([final, temp_df])

final = final.sort_values(
    ["BUY", "Strength", "P"], ascending=[False, False, False]
)
final = final[final['BUY'] == 'Yes']
final.to_csv("scan.csv", index=False)

time = pd.Timestamp.now()
time = time.floor(freq="S")
elapsed = time - time1
print("\n", time, "\n", elapsed, "\n")
