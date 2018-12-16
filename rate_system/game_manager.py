# -*- coding: utf-8 -*-
import ai_player
import utils
import random
import time
import sys
from rating import elo

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("invalid game : no user_id")
    uid = sys.argv[1]
        
    random.seed(time.time())
    
    board = utils.mock_board()
    ai = ai_player.mock_ai()

    player_color = random.choice("bw")
    if player_color == "b":
        next_move = ai.next_move()
        board.movement(next_move)
        print("Game Start. You are Black. AI moves :" + next_move)
    else:
        print("Game Start. You are White. Your turn")

    while 1:
        next_move = input()
        board.movement(next_move)

        if board.is_end():
            result = board.is_win(player_color)
            print("Game End. Winner is " + str(result))
            elo.update_rating(uid, ai.get_id(), result)
            break
    
        next_move = ai.next_move()
        board.movement(next_move)

        if board.is_end():
            result = board.is_win(player_color)
            print("AI moves :" + next_move +
                  ". Game End. Winner is " + str(result))
            elo.update_rating(uid, ai.get_id(), result)
            break
        print("AI moves :" + next_move)
    
    
