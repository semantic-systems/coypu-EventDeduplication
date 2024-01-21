from flask import Flask, jsonify, request
from flask_healthz import healthz
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
from torch import nn


app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ={
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)

model = SentenceTransformer("/data/model/")
classifier = nn.Linear(3 * 256, 2)


@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if key == 'VR9DNM826HCSUK3Q': authenticated = True

    if authenticated == False:
        response = {'error': 'no valid API key'}
        http_code = 401

    elif ('sentence_a' in request.json) and ('sentence_b' in request.json):
        sentence_a = request.json['sentence_a']
        sentence_b = request.json['sentence_b']

        # if isinstance(sentence_a, str) and isinstance(sentence_b, str):
        # inputs: List = [InputExample(texts=[sentence_a, sentence_b])]
        # elif isinstance(sentence_a, str) and isinstance(sentence_b, list):
        #     inputs: List = [InputExample(texts=[sentence_a, s]) for s in sentence_b]
        # elif isinstance(sentence_a, list) and isinstance(sentence_b, str):
        #     inputs: List = [InputExample(texts=[s, sentence_b]) for s in sentence_a]
        # elif isinstance(sentence_a, list) and isinstance(sentence_b, list):
        #     inputs: List = [InputExample(texts=[sentence_a[i], sentence_b[i]]) for i in range(len(sentence_a))]
        # else:
        #     raise TypeError(f"Type Error: Type for both instruction and sentence need to be List or String. If both are List, their length must be same.")

        inputs = [sentence_a, sentence_b]
        with torch.no_grad():
            reps = model.encode(inputs, convert_to_tensor=True)
            rep_a, rep_b = reps
            vectors_concat = []
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
            vectors_concat.append(torch.abs(rep_a - rep_b))
            features = torch.cat(vectors_concat, 0)
            output = classifier(features)
        label_mapping = ['non-dudplicate', 'duplicate']
        a = output.argmax().numpy()
        prediction = label_mapping[a]
        response = {'predictions': prediction}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    # sentence_a = "Killer winter tornado stuns storm - savvy Alabama town"
    # sentence_b = "Japan: Aftermath Of Devastating Earthquake In Noto Peninsula 14"
    # model = SentenceTransformer("/Users/hyperbolicjb/Projects/sems/sems-event-deduplication/outputs/v1/event_deduplication")
    # inputs = [sentence_a, sentence_b]#[InputExample(texts=[sentence_a, sentence_b])]
    # with torch.no_grad():
    #     reps = model.encode(inputs, convert_to_tensor=True)
    #     rep_a, rep_b = reps
    #     vectors_concat = []
    #     vectors_concat.append(rep_a)
    #     vectors_concat.append(rep_b)
    #     vectors_concat.append(torch.abs(rep_a - rep_b))
    #     features = torch.cat(vectors_concat, 0)
    #     classifier = nn.Linear(3*256, 2)
    #     output = classifier(features)
    # label_mapping = ['non-dudplicate', 'duplicate']
    # a = output.argmax().numpy()
    # # predictions = [label_mapping[score_max] for score_max in output.argmax()]
    # predictions = label_mapping[a]
    # print(predictions)