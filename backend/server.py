from flask import Flask, request, jsonify, render_template, url_for, make_response, Response, stream_with_context
import json
import subprocess
import chess
import os
import requests

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
global proc
proc = None

global info
info = {
    "player" : {},
    "computer" : {}
}

@app.route("/", methods=['GET'])
def hello():
    return render_template("index.html")

@app.route("/test/<name>", methods=['GET', 'POST'])
def test(name):
    print(request.headers)
    print(request.data)
    print(request.args)
    print(request.form)
    return "Hello %s!" % name

@app.route("/<action>", methods=['POST'])
def NUGU(action):
    print(request.headers)
    req = request.get_json()
    print(json.dumps(req, indent=4,  ensure_ascii=False))
    resp = {}
    resp["version"] = "2.0"
    resp["resultCode"] = "OK"
    resp["output"] = {}

    global proc
    
    action_name = req["action"]["actionName"]
    if action_name == "action.game.start":
        if proc == None:
            p = subprocess.Popen(['python', 'game_provider.py'],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
            proc = p
            resp["output"]["player_color"] = proc.stdout.readline().rstrip().decode()
            proc.stdin.write("ok\n".encode())
            proc.stdin.flush()
            print("GAME START")

        else:
            resp["output"]["player_color"] = "invalid"
            print("ERROR : GAME ALREADY STARTED")


    elif action_name == "action.input.move":
        if proc == None:
            resp["output"]["is_running"] = "False"
        else:

            move0 = req["action"]["parameters"]["move0"]["value"]
            move1 = req["action"]["parameters"]["move1"]["value"]
            move2 = req["action"]["parameters"]["move2"]["value"]
            move3 = req["action"]["parameters"]["move3"]["value"]

            move0 = move0.lower()
            move2 = move2.lower()

            player_move = move0 + move1 + move2 + move3
            resp["output"]["player_move"] = player_move

            proc.stdin.write((player_move+"\n").encode())
            proc.stdin.flush()

            is_valid_move = proc.stdout.readline().rstrip().decode()
            resp["output"]["is_valid_move"] = is_valid_move

            if is_valid_move == "False":
                print("ERROR : CANNOT MOVE TO "+player_move)
            else:
                print("PLAYER MOVED : "+ player_move)


    elif action_name == "action.input.valid":
        player_ann = proc.stdout.readline().rstrip().decode()
        resp["output"]["player_ann"] = player_ann
        print("PLAYER ANNOTATION : "+ player_ann)

        is_game_win = proc.stdout.readline().rstrip().decode()
        resp["output"]["is_game_win"] = is_game_win

        if is_game_win == "True":
            proc.kill()
            proc = None
            print("GAME WIN")
    
    elif action_name == "action.computer.move":
        computer_move = proc.stdout.readline().rstrip().decode()
        resp["output"]["computer_move"] = computer_move
        print("COMPUTER MOVED : "+ computer_move)
    
    elif action_name == "action.computer.annotation":
        computer_ann = proc.stdout.readline().rstrip().decode()
        resp["output"]["computer_ann"] = computer_ann
        print("COMPUTER ANNOTATION : "+ computer_ann)

        is_game_lose = proc.stdout.readline().rstrip().decode()
        resp["output"]["is_game_lose"] = is_game_lose

        if is_game_lose == "True":
            proc.kill()
            proc = None
            print("GAME LOSE")

    else:
        print("Invalid action name")
        exit(0)
        
    return jsonify(resp)

@app.route("/saveinfo", methods=['POST'])
def saveinfo():
    global info
    req = request.get_json()
    if req["player"] == "True":
        info["player"]["fen"] = req["fen"]
        info["player"]["move"] = req["move"]
        info["player"]["ann"] = req["ann"]
    else:
        info["computer"]["fen"] = req["fen"]
        info["computer"]["move"] = req["move"]
        info["computer"]["ann"] = req["ann"]
    return "OK"

@app.route("/getinfo", methods=['GET'])
def getinfo():

    return jsonify(info)


@app.route("/health", methods=['GET'])
def healthcheck():
    return "OK"


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host = '0.0.0.0', port = 5000, debug = True)
