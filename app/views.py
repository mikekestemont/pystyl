import json
import flask
from app import app
from tree import get_example_tree
from pystyl.experiment import Experiment
from collections import defaultdict

# initialize module-wide experiment in GUI-mode:
experiment = Experiment(mode='GUI')

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/import_corpus', methods=['POST'])
def import_corpus():
    if flask.request.method == 'POST':
        print('importing')
        data = {i['name']:i['value'] for i in flask.request.get_json()}
        data = defaultdict(lambda:None, data)
        experiment.import_data(directory=data['files-name'],
                               extension=data['extension-name'],
                               alpha_only=data['remove-alpha-name'],
                               lowercase=data['lowercase-name'])
        return json.dumps({"message": '<p>success</p>'})

@app.route('/preprocess_corpus', methods=['POST'])
def preprocess_corpus():
    if flask.request.method == 'POST':
        print('processing')
        data = {i['name']:i['value'] for i in flask.request.get_json()}
        data = defaultdict(lambda:None, data)
        experiment.preprocess(segment_size=int(data['segment-size-name']),
                              step_size=int(data['step-size-name']),
                              min_size=int(data['min-size-name']),
                              max_size=int(data['max-size-name']),
                              rm_tokens=data['rm-tokens-name'],
                              tokenizer_option=data['tokenizer-option-name'],
                              rm_pronouns=data['rm-pronouns-name'],
                              language=data['language-name'])
        return json.dumps({"message": '<p>success</p>'})

@app.route('/feature_extraction', methods=['POST'])
def feature_extraction():
    if flask.request.method == 'POST':
        print('features')
        data = {i['name']:i['value'] for i in flask.request.get_json()}
        data = defaultdict(lambda:None, data)
        experiment.extract_features(mfi=int(data['mfi-name']),
                                    ngram_type=data['feature-type-name'],
                                    ngram_size=int(data['ngram-size-name']),
                                    vector_space=data['vector-space-name'],
                                    min_df=float(data['min-df-name']),
                                    max_df=float(data['max-df-name']))
        return json.dumps({"message": '<p>success</p>'})

@app.route('/visualize', methods=['POST'])
def visualize():
    if flask.request.method == 'POST':
        print('visualize')
        data = {i['name']:i['value'] for i in flask.request.get_json()}
        data = defaultdict(lambda:None, data)
        svg_str = e.visualize(outputfile=data['outputfile'],
                    viz_type=data['viz-type'],
                    metric=data['metric'],
        )
        return json.dumps({"message": svg_str})
