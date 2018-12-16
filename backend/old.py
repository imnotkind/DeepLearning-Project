from flask import Flask, request, jsonify, render_template, url_for, make_response, Response, stream_with_context
import json
import subprocess
import chess
import chess.svg
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
procs = []

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
    
    action_name = req["action"]["actionName"]
    if action_name == "action.game.start":
        proc = subprocess.Popen(['python', 'game_provider.py',
                                 url_for('saveimage', _external=True)],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
        procs.append(proc)#procs[req["context"]["id"]] = proc
        resp["output"]["player_color"] = proc.stdout.readline().rstrip().decode()
        proc.stdin.write("ok\n".encode())
        proc.stdin.flush()
    elif action_name == "change.piece":
        proc = procs[0] # need to check authorization.
        piece = req["action"]["parameters"]["piece"]["value"]
        print(piece)
        proc.stdin.write(("%s\n" % piece).encode())
        proc.stdin.flush()
    else:
        proc = procs[0] # need to check authorization.
        proc.stdin.write(("%s\n" % action_name).encode())
        proc.stdin.flush()
    """
    for key, val in req["action"]["parameters"].items():
        resp["output"][key] = val["value"]"""

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
    try:
        os.remove(filename)
    except OSError:
        print('cannot remove')
    with open(filename, 'wb') as f:
        f.write(png_data)
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
