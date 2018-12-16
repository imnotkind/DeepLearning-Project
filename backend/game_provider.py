import refine_env
import argparse

if __name__ == "__main__":

    player_color = True # true is white
    cenv = refine_env.ChessEnvironment(player_color)
    if player_color == True:
        print("white") #give player color to server.py
    else:
        print("black")

    str = input()
    assert(str == "ok")

    if player_color == True: # player white : first
        while(True):
            str = input()
            (res, ann) = cenv.move_player(str)
            print(res) # is_valid_move
            if res == False:
                continue
            
            print(ann) # player_ann
            
            res = cenv.game_end()
            if res == True:
                print("win") # is_end_game
                exit(0)
            else:
                print("continue") # is_end_game

            (res, ann) = cenv.move_computer()
            print(res) # computer's move

            print(ann) # computer_ann

            res = cenv.game_end()
            if res == True:
                print("lose") # is_end_game
                exit(0)
            else:
                print("continue") # is_end_game
    
    else: # player black : second
        pass



