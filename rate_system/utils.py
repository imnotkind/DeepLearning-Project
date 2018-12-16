import random
import time

class mock_board:
    def __init__(self):
        self.state = 0
        random.seed(time.time())
    
    def get_state(self):
        return self.state

    def is_end(self):
        return self.state == 6

    def movement(self, movement=None):
        self.state += 1
        return self.state

    def is_win(self, color):
        return random.choice([1, 0.5, 0])
