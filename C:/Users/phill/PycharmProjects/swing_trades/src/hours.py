import pandas as pd
import pandas_market_calendars as mc
import schedule
import subprocess
import time
from github import Github

print("\n", "Starting the Schedule", "\n")


# main function from which other functions run
# when time and date satisfy the 'schedule' mechanism,
# this function is called to execute stock trading files
def market_hours():
    # start time is current time from which end time is determined
    start_date = pd.Timestamp.now()
    five_days = pd.Timedelta(5, "D")
    end_date = start_date + five_days

    # to avoid running the python files every day
    # and creating duplicate information, the
    # schedule will follow the NYSE's schedule
    schd = mc.get_calendar("NYSE")

    # create dataframe with start_date and end_date from above
    open_time = schd.schedule(start_date=start_date, end_date=end_date)

    # add columns to df for time in hours, minutes, and seconds
    four_hours = pd.Timedelta(4, "hours")
    eight_minutes = pd.Timedelta(8, "minutes")
    ten_seconds = pd.Timedelta(10, "seconds")
    open_time["market_open"] = open_time["market_open"] - four_hours
    open_time["market_close"] = open_time["market_close"] - four_hours
    open_time["run_code_1"] = open_time["market_close"] + ten_seconds
    open_time["run_code_1"] = open_time["run_code_1"].dt.strftime("%H:%M:%S")
    open_time["run_code_2"] = open_time["market_close"] + ten_seconds + ten_seconds
    open_time["run_code_2"] = open_time["run_code_2"].dt.strftime("%H:%M:%S")
    open_time["run_code_3"] = open_time["market_close"] + eight_minutes
    open_time["run_code_3"] = open_time["run_code_3"].dt.strftime("%H:%M:%S")
    open_time["run_code_4"] = open_time["market_close"] + eight_minutes + ten_seconds
    open_time["run_code_4"] = open_time["run_code_4"].dt.strftime("%H:%M:%S")

    print('\n', open_time, '\n')

    # save times as variables
    opening_at = open_time.iloc[0][0]
    run_the_exits = open_time.iloc[0][2]
    run_the_scan = open_time.iloc[0][3]
    run_the_positions = open_time.iloc[0][4]
    update_the_files = open_time.iloc[0][5]

    opening = pd.Timestamp(opening_at)
    todays_date = start_date.date()
    first_market_date = opening.date()

    trading_day = True if (todays_date == first_market_date) else False

    # execute the exits file
    def run_exits():
        subprocess.run(["python", "exits.py"])
        return schedule.CancelJob

    # execute the scan file
    def run_scan():
        subprocess.run(["python", "scan.py"])
        return schedule.CancelJob

    # execute the positions file
    def run_positions():
        subprocess.run(["python", "positions.py"])
        return schedule.CancelJob

    # update github
    def update_github():
        g = Github('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')
        repo = g.get_user('p-norris').get_repo('swing_trades')

        contents_h = repo.get_contents('src/history.csv')
        contents_p = repo.get_contents('src/positions.csv')

        his = 'C:/Users/phill/PycharmProjects/swing_trades/src/history.csv'
        pos = 'C:/Users/phill/PycharmProjects/swing_trades/src/positions.csv'

        with open(his) as file:
            new_h = file.read()
        with open(pos) as file:
            new_p = file.read()

        repo.update_file(his, "updating file", new_h, contents_h.sha, branch='main')
        repo.update_file(pos, "updating file", new_p, contents_p.sha, branch='main')
        return schedule.CancelJob

    # if the market is open, schedule runs all the files
    # positions runs 8 minutes after the scan which takes a while
    # and the interface updates 1/10 of a minute later
    if trading_day:
        schedule.every().monday.at(run_the_exits).do(run_exits)
        schedule.every().monday.at(run_the_scan).do(run_scan)
        schedule.every().monday.at(run_the_positions).do(run_positions)
        schedule.every().monday.at(update_the_files).do(update_github)

        schedule.every().tuesday.at(run_the_exits).do(run_exits)
        schedule.every().tuesday.at(run_the_scan).do(run_scan)
        schedule.every().tuesday.at(run_the_positions).do(run_positions)
        schedule.every().tuesday.at(update_the_files).do(update_github)

        schedule.every().wednesday.at(run_the_exits).do(run_exits)
        schedule.every().wednesday.at(run_the_scan).do(run_scan)
        schedule.every().wednesday.at(run_the_positions).do(run_positions)
        schedule.every().wednesday.at(update_the_files).do(update_github)

        schedule.every().thursday.at(run_the_exits).do(run_exits)
        schedule.every().thursday.at(run_the_scan).do(run_scan)
        schedule.every().thursday.at(run_the_positions).do(run_positions)
        schedule.every().thursday.at(update_the_files).do(update_github)

        schedule.every().friday.at(run_the_exits).do(run_exits)
        schedule.every().friday.at(run_the_scan).do(run_scan)
        schedule.every().friday.at(run_the_positions).do(run_positions)
        schedule.every().friday.at(update_the_files).do(update_github)


# master schedule that starts the process
schedule.every().monday.at("09:15").do(market_hours)
schedule.every().tuesday.at("09:15").do(market_hours)
schedule.every().wednesday.at("09:15").do(market_hours)
schedule.every().thursday.at("09:15").do(market_hours)
schedule.every().friday.at("09:15").do(market_hours)

while True:
    schedule.run_pending()
    time.sleep(1)
