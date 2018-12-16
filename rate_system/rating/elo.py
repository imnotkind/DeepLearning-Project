import pandas as pd
import numpy as np

"""
reference from
https://math.stackexchange.com/questions/1358077/chess-rating-calculating-algorithm
"""
RATE_DB_FILE = "db.csv"
SCORE_TABLE = np.array([-800, -677, -589, -538, -501, -470, -444, -422, -401,
                        -383, -366, -351, -336, -322, -309, -296, -284, -273,
                        -262, -251, -240, -230, -220, -211, -202, -193, -184,
                        -175, -166, -158, -149, -141, -133, -125, -117, -110,
                        -102, -95, -87, -80, -72, -65, -57, -50, -43, -36,
                        -29, -21, -14, -7, 0, 7, 14, 21, 29, 36,
                        43, 50, 57, 65, 72, 80, 87, 95, 102, 110,
                        117, 125, 133, 141, 149, 158, 166, 175, 184,
                        193, 202, 211, 220, 230, 240, 251, 262, 273,
                        284, 296, 309, 322, 336, 351, 366, 383, 401,
                        422, 444, 470, 501, 538, 589, 677, 800])


def initialize():
    df = pd.DataFrame({'id':["ai0"],
                       'rating':[1000],
                       'num_games':[0]}).set_index('id')

    df.to_csv(RATE_DB_FILE, encoding="utf-8", mode='w', header=True, index=True)

def expected_score(dp):
    return SCORE_TABLE.searchsorted(dp) * 0.01
    

def k_factor(rating, num_games, age=20):
    if num_games < 30 and rating < 2300:
        return 40
    elif rating < 2400:
        return 20
    else:
        return 10

def update_rating(uid1, uid2, result):
    df = pd.DataFrame.from_csv(RATE_DB_FILE, index_col='id')
    if uid1 not in df.index:
        df.loc[uid1, ['rating', 'num_games']] = [1000, 0]
    if uid2 not in df.index:
        df.loc[uid2, ['rating', 'num_games']] = [1000, 0]


    r1, n1 = df.loc[uid1, ['rating', 'num_games']]
    r2, n2 = df.loc[uid2, ['rating', 'num_games']]

    p = expected_score(r1 - r2)
    k1 = k_factor(r1, n1)
    k2 = k_factor(r2, n2)

    r1 += k1 * (result - p)
    r2 += k2 * (-result + p)
    df.loc[uid1, ['rating', 'num_games']] = [r1, n1 + 1]
    df.loc[uid2, ['rating', 'num_games']] = [r2, n2 + 1]

    df.to_csv(RATE_DB_FILE, encoding="utf-8", mode='w', header=True, index=True)
    return r1, r2

def get_rating(uid):
    df = pd.DataFrame.from_csv(RATE_DB_FILE, index_col='id')
    if uid not in df.index:
        df.loc[uid] = [1000, 0]
        df.to_csv(RATE_DB_FILE, encoding="utf-8", mode='w', header=True, index=True)
    return df.loc[uid, 'rating']

    

if __name__ == "__main__":
    initialize()
