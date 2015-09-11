import json
import flask
from app import app
from tree import get_example_tree

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template("index.html")

@app.route('/processResults', methods=["POST"])
def processResults():
    print flask.request.method
    if flask.request.method == 'POST':
        data = flask.request.json
        return json.dumps({"message": get_example_tree()})
    # return flask.redirect("index")
