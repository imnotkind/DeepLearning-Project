from flask import Flask, request, jsonify
import json

app = Flask(__name__)

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
    for key, val in req["action"]["parameters"].items():
        resp["output"][key] = val["value"]
    #resp["time"] = "12"
    resp["output"]["time"] = "12"
    return jsonify(resp)

@app.route("/health", methods=['GET'])
def healthcheck():
    return "OK"


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000, debug = True)

