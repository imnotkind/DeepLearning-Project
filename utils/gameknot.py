import requests
from bs4 import BeautifulSoup
import re
import string
from collections import defaultdict
import operator
import pandas as pd

class GameKnot(object):

    def __init__(self):
        self.host = 'gameknot.com'
        self.meaningless_words = ['i', 'he', 'a', 'the', 'if', 'and', 'or', 'is', 'was', 'this', 'that', 'when']

    def get_url(self, path):
        assert path.startswith('/')
        return 'https://' + self.host + path

    def get_annotated_games(self):
        games = []
        for i in range(308):
            url = self.get_url('/list_annotated.pl?u=all')
            params = {'p': i}
            html = requests.get(url, params=params).content

            _all = BeautifulSoup(html, 'html.parser')
            table_rows = _all.select('.evn_list,.odd_list')
            links = list(map(lambda e: e.select('a')[1]['href'], table_rows))
            games += links

        return games

    def get_game_infos(self, game):
        game_infos = []
        
        url = self.get_url(game)
        html = requests.get(url).content

        turn_infos = self.get_turn_infos(html)
        annotations = self.get_annotations(html)
        assert len(turn_infos) == len(annotations)

        for i in xrange(len(turn_infos)):
            [fen, move] = turn_infos[i]
            game_infos.append([fen, move, annotations[i]])

        return game_infos

    def get_annotations(self, html):
        _all = BeautifulSoup(html, 'html.parser')
        annotations = _all.select('.dialog')[0].select('td[style="vertical-align: top;"]')
        annotations = list(map(lambda e: str(e.text.encode('utf8')), annotations))
        return annotations

    def get_turn_infos(self, html):
        turn_infos = re.findall("render_chess_diagram\('(.*)'\)", html)
        turn_infos = list(map(lambda e: e.split('|||last='), turn_infos))
        return turn_infos

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

        # print self.get_freq(words)

        return words

    def get_freq(self, words):
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
        return sorted(freq.items(), key=operator.itemgetter(1), reverse=True)


def main():
    gk = GameKnot()

    games = gk.get_annotated_games()
    print(len(games))
    exit()

    data = list()
    
    for i, game in enumerate(games):
        game_infos = gk.get_game_infos(game)
        for j, game_info in enumerate(game_infos):
            # print '-' * 100
            # print game_info

            [fen, move, annotation] = game_info
            words = gk.get_words(annotation)

            #raw_input()

            data.append([i, j, fen, move, annotation])
        print('[%d / %d]' % (i+1, len(games)))

    '''
    df = pd.DataFrame(data, columns=['gid', 'aid', 'fen', 'move', 'annotation'])
    df.to_csv("gameknot.csv", mode='w')
    '''
    
if __name__ == '__main__':
    main()
