from flask import Flask
from flask import request

import core
net = core.load()["net"]

app = Flask(__name__)
@app.route("/")
def hello():
    seed = request.args.get('seed', 'The ')
    n = int(request.args.get('n', "200"))
    return net.predict(seed, seq_len=n)