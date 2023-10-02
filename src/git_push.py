from github import Github

g = Github('code')
repo = g.get_user('p-norris').get_repo('swing_trades')

contents_h = repo.get_contents('src/history.csv')
contents_p = repo.get_contents('src/positions.csv')

with open('C:/Users/phill/PycharmProjects/swing_trades/src/history.csv') as file:
    new_h = file.read()
with open('C:/Users/phill/PycharmProjects/swing_trades/src/positions.csv') as file:
    new_p = file.read()

repo.update_file('history.csv', "updating file", new_h, contents_h.sha, branch='main')
repo.update_file('positions.csv', "updating file", new_p, contents_p.sha, branch='main')
