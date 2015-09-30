import json
import flask
from app import app
from tree import get_example_tree
from pystyl.experiment import Experiment

# initialize module-wide experiment in GUI-mode:
experiment = Experiment(mode='GUI')

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/processResults', methods=['POST'])
def processResults():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        print(data)
        print(experiment)
        return json.dumps({"message": get_example_tree()})

@app.route('/import_corpus', methods=['POST'])
def import_corpus():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        experiment.import_data()

@app.route('/preprocess_corpus', methods=['POST'])
def preprocess_corpus():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        experiment.import_data()

@app.route('/feature_extraction', methods=['POST'])
def feature_extraction():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        experiment.extract_features()

@app.route('/visualize', methods=['POST'])
def visualize():
    if flask.request.method == 'POST':
        data = flask.request.get_json()
        svg_str = experiment.visualize()
        return json.dumps({"message": svg_str})
