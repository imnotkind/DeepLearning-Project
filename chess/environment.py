from sunfish import *
from sunfish import Position as SunPosition
from tools import renderFEN
import copy
import re
import random
import signal
import requests
from time import sleep

def input_with_timeout(timeout):
    signal.signal(signal.SIGALRM, lambda: (_ for _ in ()).throw(Exception('timeout')))
    signal.alarm(timeout)

    try:
        return input()
    finally:
        signal.alarm(0)

class Position(SunPosition):
    """ For taking attention to a specific piece by overriding,
    define a child class. """
    def __new__(cls, board, score, wc, bc, ep, kp, piece=None, rs=[]):
        self = super(Position, cls).__new__(cls, board, score, wc, bc, ep, kp)
        self.piece = piece
        self.rs = rs
        """
        def dec(sun_func):
            def wrapper(*args):
                sun_pos = sun_func(*args)
                return Position.fromSunPos(sun_pos, piece=piece, rs=rs)
            return wrapper

        self.rotate = dec(super(Position, self).rotate)
        self.nullmove = dec(super(Position, self).nullmove)"""
        
        return self

    @classmethod
    def fromSunPos(cls, sun_pos, piece=None, rs=[]):
        if isinstance(sun_pos, SunPosition):
            pos = copy.copy(sun_pos)
            pos.piece = piece
            pos.rs = rs
            pos.__class__ = Position
            return pos

    def focus(self, piece=None, rs=[]):
        return Position(self.board, self.score, self.bc, self.wc, self.ep, self.kp,
                        piece=piece, rs=rs)
        
    def gen_moves(self):
        # For any specific piece (e.g. PNBRQK), iterate like gen_moves in parent class.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            if self.piece and p != self.piece: continue ###
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp): break
                    # Move it
                    if (i, j) not in self.rs:
                        yield (i, j) 
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0] and (j+E, j+W) not in self.rs: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1] and (j+W, j+E) not in self.rs: yield (j+W, j+E)

    def move(self, move):
        sun_pos = super(Position, self).move(move)
        return Position.fromSunPos(sun_pos, piece=self.piece, rs=self.rs)

    def rotate(self):
        sun_pos = super(Position, self).rotate()
        return Position.fromSunPos(sun_pos, piece=self.piece, rs=self.rs)

    def nullmove(self):
        sun_pos = super(Position, self).nullmove()
        return Position.fromSunPos(sun_pos, piece=self.piece, rs=self.rs)
class ChessEnvironment:
    """ A chess environment
    pos : a current board and pieces
    ps : player's ai
    os : opponent ai
    pl : player ai's level
    ol : opponent ai's level
    pc : player's color"""
    
    def __init__(self, pl=0.001, ol=0.5, timeout=3):
        self.pos = Position(initial, 0, (True,True), (True,True), 0, 0)
        self.ps = Searcher()
        self.os = Searcher()
        self.pl = pl
        self.ol = ol
        self.auto = False
        self.timeout = timeout

        # player's color
        if random.random() > 0.5 :
            self.pc = "white"
        else:
            self.pc = "black"

    def propose_move(self, piece=None, rs=[]):
        if self.pos.score <= -MATE_LOWER:
            return "lose"
        if not piece and len(rs) == 0:
            move, _ = self.ps.search(self.pos, secs=self.pl)
        else:
            move, _ = self.ps.search(self.pos.focus(piece, rs), secs=self.pl)
        return move

    def auto_move(self, lastmove, url):
        rs = []
        piece = None
        while True:
            #try:
            move = self.propose_move(piece, rs)
            #except:
            #    move = self.propose_move()
            nextmove = render(move[0]) + render(move[1])# + ',' + render(move[0])
            self.make_request(url, nextmove=nextmove+','+render(move[0]),
                              lastmove=lastmove)
            try:
                response = input_with_timeout(self.timeout)
            except:
                response = 'ok'

            if response == 'ok':
                break
            elif response == 'change.move':
                self.ps.tp_move.__init__(TABLE_SIZE)
                self.ps.tp_score.__init__(TABLE_SIZE)
                rs.append(move)
            elif response in "PNBRQK":
                self.ps.tp_move.__init__(TABLE_SIZE)
                self.ps.tp_score.__init__(TABLE_SIZE)
                rs.append(move)
                piece = response
            elif response == "faster":
                timeout=2
                secs=1
            elif response == "stop":
                pass

        return move, nextmove

    def process(self, url=None, secs=3):
        lastmove = None
        
        if self.pc == "black":
            self.pos = self.pos.rotate()
        self.make_request(url)

        if self.pc == "black":
            sleep(secs)
            self.pos = self.pos.rotate()
            opp_move, score = self.os.search(self.pos, secs=self.ol)

            self.pos = self.pos.move(opp_move)
            lastmove = render(119-opp_move[0]) + render(119-opp_move[1])
            self.make_request(url, lastmove=lastmove)

        while True:
            sleep(secs)
            
            if self.auto:
                move, nextmove = self.auto_move(lastmove, url)
                lastmove=nextmove
            else:
                move = input()
                match = re.match('([a-h][1-8])'*2, move)
                if match:
                    print("moved")
                else:
                    print("invalid")
            
            self.pos = self.pos.move(move)
            self.pos = self.pos.rotate()
            self.make_request(url, lastmove=lastmove)
            self.pos = self.pos.rotate()
            sleep(secs)

            if self.pos.score <= -MATE_LOWER:
                return "win"

            opp_move, score = self.os.search(self.pos, secs=self.ol)

            if score == MATE_UPPER:
                return "lose"

            self.pos = self.pos.move(opp_move)
            lastmove = render(119-opp_move[0]) + render(119-opp_move[1])
            self.make_request(url, lastmove=lastmove)


    def make_request(self, url, lastmove='', nextmove=''):
        fen = renderFEN(self.pos).split()[0]
        if lastmove is None:
            lastmove = ''
        if nextmove is None:
            nextmove = ''
        if self.pc == "black":
            fen = fen[::-1]
        
        try:
            response = requests.post(url, json={'fen' : fen,
                                                'lastmove' : lastmove,
                                                'arrows' : nextmove})
        except:
            response = None
        return response
    
    def reset(self, pl, ol):
        self.position = Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self.pl = pl
        self.ol = ol

    def state(self):
        return self.pos.board, self.pos.score

    def nugu_move(self, move):
        pass
    
    def move(self, move):
        if self.position.score <= -sunfish.MATE_LOWER:
            return "lose"
        
        match = re.match('([a-h][1-8])'*2, move)
        if match:
            move = sunfish.parse(match.group(1)), sunfish.parse(match.group(2))
            self.position = self.position.move(move)

            if self.position.score <= -sunfish.MATE_LOWER:
                return "win"

            oppMove, score = self.searcher.search(self.position, secs=(self.level * 0.1))

            if score == sunfish.MATE_UPPER:
                return "lose"

            self.position = self.position.move(move)
            return sunfish.render(119-move[0]) + sunfish.render(119-move[1])
        else:
            return "invalid format"


if __name__ == "__main__":
    cenv = ChessEnvironment()
    
