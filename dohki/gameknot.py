import requests
from bs4 import BeautifulSoup
import re
import string
from collections import defaultdict
import operator
import traceback
import chess
import pandas as pd
import sys
from multiprocessing.dummy import Pool as ThreadPool
import time


class GameKnot(object):

    def __init__(self, is_debug=False):
        self.is_debug = is_debug
        
        self.host = 'gameknot.com'
        self.list_url = self.get_url('/list_annotated.pl')
        self.meaningless_words = ['i', 'he', 'a', 'the', 'if', 'and', 'or', 'is', 'was', 'this', 'that', 'when']

    def get_url(self, path):
        assert path.startswith('/')
        return 'https://' + self.host + path

    def get_page(self, url, params):
        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
        html = requests.get(self.list_url, params=params, headers=headers).content
        return html

    '''
    # for loop for the responsiveness
    def get_data(self):
        params = {'u': 'all'}
        html = requests.get(self.list_url, params=params).content

        pg_num = self.get_pg_num(html)
        # TODO
        pg_num = 16
        pgs = [i for i in range(pg_num)]

        pool = ThreadPool(16)
        results = pool.map(self.do_for_page, pgs)
        pool.close()
        pool.join()

        data = [y for x in results for y in x]

        return data
    '''

    def do_for_page(self, pg):
        data = []

        params = {'u': 'all', 'p': pg-1}
        html = self.get_page(self.list_url, params)

        _all = BeautifulSoup(html, 'html.parser')
        table_rows = _all.select('.evn_list,.odd_list')
        games = list(map(lambda e: e.select('a')[1]['href'], table_rows))

        len_games = len(games)
        for i, game in enumerate(games):
            print('p.{} #{}'.format(pg, i+1))
            game_infos = self.get_game_infos(game)
            for j, game_info in enumerate(game_infos):
                if self.is_debug:
                    print('-' * 100)
                    print(game_info)
                    input()

                data.append([i, j, *game_info])
                time.sleep(0.05)

        return data

    def get_game_infos(self, game):
        game_infos = []
        
        url = self.get_url(game)
        html = requests.get(url).content

        pg_num = self.get_pg_num(html)
        for pg in range(pg_num):
            params = {'pg': pg}
            html = requests.get(url, params=params).content
            
            fens = self.get_fens(html)
            moves = self.get_moves(html)
            annotations = self.get_annotations(html)
            assert len(fens) == len(moves) == len(annotations)

            for i in range(len(fens)):
                game_infos.append([moves[i], fens[i], annotations[i]])

        return game_infos

    def get_pg_num(self, html):
        _all = BeautifulSoup(html, 'html.parser')
        pgs = _all.select('.paginator')
        
        if not pgs:
            return 1
        else:
            pgs = pgs[0].select('td')
            return int(pgs[-2].text)

    def get_fens(self, html):
        turn_infos = re.findall(b"render_chess_diagram\('(.*last.*)'\)", html)
        fens = list(map(lambda e: e.split(b'|||last=')[0], turn_infos))
        return fens

    def get_moves(self, html):
        _all = BeautifulSoup(html, 'html.parser')
        moves = _all.select('.dialog')[0].select('td[style="vertical-align: top; width: 20%;"]')
        moves = map(lambda e: e.text.splitlines()[1].replace('\xa0', ''), moves)
        moves = map(lambda e: re.sub(r'(\d\.{1,3})', '', e).split()[-1], moves)
        return list(moves)

    def get_annotations(self, html):
        _all = BeautifulSoup(html, 'html.parser')
        annotations = _all.select('.dialog')[0].select('td[style="vertical-align: top;"]')
        annotations = list(map(lambda e: str(e.text).replace(',', ''), annotations))
        return annotations

    def get_words(self, annotation):
        for ch in string.punctuation:
            annotation = annotation.replace(ch, '')

        words = annotation.split()
        words = map(lambda e: e.lower(), words)

        for meaningless_word in self.meaningless_words:
            try:
                while True:
                    words.remove(meaningless_word)
            except ValueError:
                pass

        print(self.get_freq(words))

        return words

    def get_freq(self, words):
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
        return sorted(freq.items(), key=operator.itemgetter(1), reverse=True)


def main():
    is_debug = (len(sys.argv) > 1 and sys.argv[1] == '-d')
    if is_debug:
        print('[*] DEBUG MODE')

    gk = GameKnot(is_debug)
    for i in range(308, 309):
        data = gk.do_for_page(i)

        df = pd.DataFrame(data, columns=['gid', 'aid', 'move', 'fen', 'annotation'])
        df.to_csv("gameknot_p{}.csv".format(i), mode='w')

    print('\a')


if __name__ == '__main__':
    main()
