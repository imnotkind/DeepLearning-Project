import sunfish
import re
import chess
import json
from pwn import *


class ChessEnvironment:

    def __init__(self, white=True):
        self.pos = sunfish.Position(
            sunfish.initial, 0, (True, True), (True, True), 0, 0)
        self.searcher = sunfish.Searcher()
        self.isPlayerTurn = white
        self.isPlayerwhite = white

        self.board = chess.Board()
        context.log_level = 'error'

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


        annotation = self.get_kor_sentence(uci_move)

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

        annotation = self.get_kor_sentence(uci_move)

        return (move, annotation)

    def get_kor_sentence(self, uci_move):
        san_move = self.board.san(self.board.parse_uci(uci_move))
        self.board.push_uci(uci_move)
        fen = self.board.fen()

        data = {'fen': fen, 'move': san_move}
        with remote('localhost', 51119) as r:
            r.sendline(json.dumps(data))
            return r.recvline().strip().decode()


def main():
    cenv = ChessEnvironment(True)
    cenv.move_player('e2e4')
    cenv.move_computer()


if __name__ == '__main__':
    main()
