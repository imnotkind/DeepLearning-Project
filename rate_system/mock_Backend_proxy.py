# -*- coding: utf-8 -*-
"""
import requests
r = requests.get()
r.json
"""
import json
import subprocess

if __name__ == "__main__":
    proc_dict = {}
    
    while 1:
        # simulate the requests from REST API
        request = input("request:")
        request = json.loads(request)

        uid = request['accessToken']
        action = request['action']
        # classify actions from requests
        if action == "start_game":
            proc = subprocess.Popen(['python', 'game_manager.py', uid],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
            proc_dict[uid] = proc
            print("response: " + proc.stdout.readline().rstrip().decode())
    
        elif "move" in action:
            cur, next = action.split()[1:]
            next_move = 'from {} to {}\n'.format(cur, next)
            
            proc = proc_dict[uid]
            proc.stdin.write(next_move.encode())
            proc.stdin.flush()
            print("response: " + proc.stdout.readline().rstrip().decode())
    
