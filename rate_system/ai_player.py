import random

def random_an():
    """ an random algebraic notation generator"""
    alpha = "abcdefgh"
    number = "12345678"
    return random.choice(alpha) + random.choice(number)

class mock_ai:
    def __init__(self):
        self.id = "ai0"
    def next_move(self, state=None):
        return "from " + random_an() + " to " + random_an()
    def get_id(self):
        return self.id
