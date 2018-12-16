from flask import Flask, request, jsonify, render_template, url_for, make_response, Response, stream_with_context
import json
import subprocess
import chess
import chess.svg
import cairosvg
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
global proc
proc = None

@app.route("/", methods=['GET'])
def hello():
    print(request.headers)
    print(request.get_json())
    return "Hello World!"

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
            p = subprocess.Popen(['python', 'game_provider.py',
                                    url_for('saveimage', _external=True)],
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
            resp["output"]["is_running"] = "false"
        else:

            move0 = req["action"]["parameters"]["move0"]["value"]
            move1 = req["action"]["parameters"]["move1"]["value"]
            move2 = req["action"]["parameters"]["move2"]["value"]
            move3 = req["action"]["parameters"]["move3"]["value"]

            move0 = move0.lower()
            move2 = move2.lower()

            move = move0 + move1 + move2 + move3

            proc.stdin.write((move+"\n").encode())
            proc.stdin.flush()

            print("PLAYER MOVED : "+ move)

            is_valid_move = proc.stdout.readline().rstrip().decode()
            resp["output"]["is_valid_move"] = is_valid_move

            if is_valid_move == "False":
                print("ERROR : CANNOT MOVE TO "+move)
                return jsonify(resp)

            is_end_game = proc.stdout.readline().rstrip().decode()
            resp["output"]["is_end_game"] = is_end_game

            if is_end_game != "continue":
                proc.kill()
                proc = None
                print("GAME END : "+ is_end_game)
                return jsonify(resp)

            computer_move = proc.stdout.readline().rstrip().decode()
            resp["output"]["computer_move"] = computer_move

            print("COMPUTER MOVED : "+ computer_move)
            
            is_end_game = proc.stdout.readline().rstrip().decode()
            resp["output"]["is_end_game"] = is_end_game

            if is_end_game != "continue":
                proc.kill()
                proc = None
                print("GAME END : "+ is_end_game)
                return jsonify(resp)


    else:
        print("Invalid action name")
        exit(0)
        
    return jsonify(resp)

@app.route("/image", methods=['GET'])
def image():
    board_image = request.args.get('path', default='static/board0.png', type=str)
    def generate():
        while True:
            frame = open(board_image, 'rb').read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
    return Response((generate()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/saveimage", methods=['POST'])
def saveimage():
    req = request.get_json()
    fen = req.get('fen', '')
    if fen is "":
        return "invalid"
    try:
        lastmove = chess.Move.from_uci(req.get('lastmove', None))
    except:
        lastmove = None
    arrows = req.get('arrows', "")

    board = chess.BaseBoard("/".join(fen.split("/")[0:8]))
    arrows = [arrow(s.strip()) for s in arrows.split(',') if s.strip()]
    svg_data = chess.svg.board(board, coordinates=False, flipped=False, lastmove=lastmove, check=None, arrows=arrows, size=360, style=None)
    png_data = cairosvg.svg2png(bytestring=svg_data)
    filename = 'static/board0.png'
    with open(filename, 'wb') as f:
        f.write(png_data)
    print("SAVED on"+filename)
    return "rendering image saved"

def arrow(s):
    tail = chess.SQUARE_NAMES.index(s[:2])
    head = chess.SQUARE_NAMES.index(s[2:]) if len(s) > 2 else tail
    return chess.svg.Arrow(tail, head)

@app.route("/health", methods=['GET'])
def healthcheck():
    return "OK"


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host = '0.0.0.0', port = 5000, debug = True)
