import sunfish
import re
import chess
import json
from pwn import *
import requests


class ChessEnvironment:

    def __init__(self, white=True):
        self.pos = sunfish.Position(
            sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self.searcher = sunfish.Searcher()
        self.isPlayerTurn = white
        self.isPlayerwhite = white

        self.board = chess.Board()
        context.log_level = 'error'

        data = {'fen': "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", 'move': "Start", 'player': "True", 'ann': "None"}
        r = requests.post("http://141.223.163.184:5000/saveinfo", json=data)

    def reset(self, white=True):
        self.pos = sunfish.Position(
            sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self.isPlayerTurn = white
        self.isPlayerwhite = white

        self.board = chess.board()

    def state(self):
        return self.pos.board if self.isPlayerTurn else self.pos.rotate().board

    def game_end(self):
        if self.isPlayerTurn and self.pos.score <= -sunfish.MATE_LOWER:
            return True
        if not self.isPlayerTurn and self.pos.score <= -sunfish.MATE_LOWER:
            return True
        return False

    def move_player(self, move):
        uci_move = move

        if not self.isPlayerTurn:
            return (False, None)
        if self.game_end():
            return (False, None)
        match = re.match('([a-h][1-8])'*2, move)
        if match:
            if self.isPlayerwhite:
                move = sunfish.parse(match.group(1)), sunfish.parse(match.group(2))
            else:
                move = 119 - sunfish.parse(match.group(1)), 119 - sunfish.parse(match.group(2))
        else:
            return (False, None)
        if move not in self.pos.gen_moves():
            return (False, None)
        self.pos = self.pos.move(move)


        annotation = self.get_kor_sentence(uci_move, True)

        self.isPlayerTurn = False
        return (True, annotation)

    def move_computer(self):
        move, score = self.searcher.search(self.pos, secs=0.5)
        self.pos = self.pos.move(move)

        self.isPlayerTurn = True
        if self.isPlayerwhite:
            uci_move = sunfish.render(119-move[0]) + sunfish.render(119-move[1])
        else:
            uci_move = sunfish.render(move[0]) + sunfish.render(move[1])

        annotation = self.get_kor_sentence(uci_move, False)

        return (uci_move, annotation)

    def get_kor_sentence(self, uci_move, player):
        try:
            san_move = self.board.san(self.board.parse_uci(uci_move))
            self.board.push_uci(uci_move)
            fen = self.board.fen()
        except:
            ann = "그는 말이 없다."
            data2 = {'fen': "None", 'move': uci_move, 'player': str(player), 'ann': ann}
            r = requests.post("http://141.223.163.184:5000/saveinfo", json=data2)
            return ann
        

        data = {'fen': fen, 'move': san_move}
        with remote('localhost', 51118) as r:
            r.sendline(json.dumps(data))
            ann = r.recvline().strip().decode()

  
            data2 = {'fen': fen, 'move': uci_move, 'player': str(player), 'ann': ann}
            r = requests.post("http://141.223.163.184:5000/saveinfo", json=data2)

            return ann


def main():
    cenv = ChessEnvironment(True)
    cenv.move_player('e2e4')
    cenv.move_computer()


if __name__ == '__main__':
    main()
