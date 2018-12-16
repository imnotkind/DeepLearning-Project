import sunfish
import re

class ChessEnvironment:

	def __init__(self, white = True):
		self.pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
		self.searcher = sunfish.Searcher()
		self.isPlayerTurn = white
		self.isPlayerwhite = white

	def reset(self, white = True):
		self.pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
		self.isPlayerTurn = white
		self.isPlayerwhite = white

	def state(self):
		return self.pos.board if self.isPlayerTurn else self.pos.rotate().board

	def game_end(self):
		if self.isPlayerTurn and self.pos.score <= -sunfish.MATE_LOWER:
			return True
		if not self.isPlayerTurn and self.pos.score <= -sunfish.MATE_LOWER:
			return True
		return False

	def move_player(self, move):
		if not self.isPlayerTurn:
			return False
		if self.game_end():
			return False
		match = re.match('([a-h][1-8])'*2, move)
		if match:
			if self.isPlayerwhite:
				move = sunfish.parse(match.group(1)), sunfish.parse(match.group(2))
			else:
				move = 119 - sunfish.parse(match.group(1)), 119 - sunfish.parse(match.group(2))
		else:
			return False
		if move not in self.pos.gen_moves():
			return False
		self.pos = self.pos.move(move)
		self.isPlayerTurn = False
		return True

	def move_computer(self):
		move, score = self.searcher.search(self.pos, secs=0.5)
		self.pos = self.pos.move(move)
		self.isPlayerTurn = True
		if self.isPlayerwhite:
			return sunfish.render(119-move[0]) + sunfish.render(119-move[1])
		else:
			return sunfish.render(move[0]) + sunfish.render(move[1])

