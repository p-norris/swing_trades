import pandas as pd
import yfinance as yf

# for terminal, signal beginning of script with time
print("\n", "Exits", "\n")
time1 = pd.Timestamp.now()
time1 = time1.floor(freq="S")
print(time1, '\n')

# read in the relevant datasets
df_positions = pd.read_csv("positions.csv")
df_history = pd.read_csv("history.csv")
df_all_sectors = pd.read_csv('tickers_sectors_6000.csv')

if len(df_positions) == 0:
    exit("Got nothing to trade. Must be a bad market.")

# establish necessary variables
positions = df_positions["STOCK"].tolist()  # list of stocks to sell

timed_out = df_positions.iloc[0]['Sell On 2']  # sell date when target isn't reached

balance = df_history.iloc[0]["BALANCE"]  # get balance for updating sells

sectors_dict = dict(zip(df_all_sectors.SYMBOL,  # create dict of sectors for dashboard
                        df_all_sectors.SECTOR))

current_day = pd.Timestamp.now()  # current date for sells / trade history

# loop through each of the stocks in the positions list
# to determine trade and value, then record the exchange
# in the history and positions dataframes
# declare stock variable to use when assigning values
# from a particular 'stock' from the positions list
stock = 0
for position in positions:

    # saves trade data in row format to add to history file
    def trade_record(sell_price, balance):
        buy_date = df_positions.iloc[stock]["Buy Date"]
        sell_date = pd.Timestamp.now()
        share_price = df_positions.iloc[stock]["Share Price"].round(2)
        shares = df_positions.iloc[stock]["Shares"]
        cost = df_positions.iloc[stock]["Cost"].round(2)
        sell_price = sell_price.round(2)
        income = (shares * sell_price).round(2)
        profit = (income - cost).round(2)
        pct_chg = ((profit / cost) * 100).round(2)
        balance = (balance + profit).round(2)
        nuro = [position,
                sectors_dict[position],
                buy_date,
                sell_date,
                shares,
                share_price,
                cost,
                sell_price,
                income,
                profit,
                pct_chg,
                balance]
        return nuro


    # updates the positions and history dataframes
    def update_datasets(nuro, df_history, df_positions):
        df_history.loc[-1] = nuro
        df_history.index = df_history.index + 1
        df_history.sort_index(inplace=True)
        df_positions = df_positions.drop(df_positions.index[stock])
        df_positions = df_positions.reset_index(drop=True)
        nuro.clear()
        return df_history, df_positions


    # get trade history for one day per minute from yahoo!
    df = yf.download(position, period="1d", interval="1m")

    # save to variables the trade prices
    high_price = df_positions.iloc[stock]["Target Price"]
    low_price = df_positions.iloc[stock]["Hit Price"]

    # assign False to 'sold' variable so that 'sell_price'
    # does not incorrectly update if already assigned
    sold = False

    # result = hi_low(df)
    for i in range(len(df)):
        if df.iloc[i]["Close"] >= high_price:
            sell_price = df.iloc[i]["Close"]
            nuro = trade_record(sell_price, balance)
            df_history, df_positions = update_datasets(nuro, df_history, df_positions)
            sold = True
            print('\nSold High\n')
            break
        elif i > 33:
            if df.iloc[i]["Close"] <= low_price:
                sell_price = df.iloc[i]["Close"]
                nuro = trade_record(sell_price, balance)
                df_history, df_positions = update_datasets(nuro, df_history, df_positions)
                sold = True
                print('\nSold Low\n')
                break

    # if position does not meet either trade criteria,
    # sell it at the end of the second day, then update
    # the 'stock' variable for the next position
    if (sold is False) & (current_day == timed_out):
        row = len(df)
        sell_price = df.iloc[row]['Close']
        nuro = trade_record(sell_price, balance)
        df_history, df_positions = update_datasets(nuro, df_history, df_positions)
        print('\nTimed Out\n')
    elif (sold is False) & (current_day != timed_out):
        print('\nNEXT\n')
        stock += 1

# normalize the dates so that they are properly formatted for future use
df_history['Buy Date'] = pd.DatetimeIndex(df_history['Buy Date']).normalize()
df_history['Sell Date'] = pd.DatetimeIndex(df_history['Sell Date']).normalize()

# save the position and history dataframes to csv files
df_positions.to_csv("positions.csv", index=False)
df_history.to_csv("history.csv", index=False)

# update the terminal with status information
print("\nHistory:\n", df_history)

print("\n*** Positions: ***\n", df_positions, "\n") if len(df_positions) > 0 else print(
    "\n No current positions")

time = pd.Timestamp.now()
time = time.floor(freq="S")
elapsed = time - time1
print("\n", time, "\n", elapsed, "\n")
