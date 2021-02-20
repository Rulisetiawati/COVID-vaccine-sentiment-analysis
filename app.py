from flask import Flask, render_template, url_for, request
from deployment_model.seq_model import SeqModel
from preprocessing_helper import *
from torchtext.data import Field, Pipeline
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS

import torch
import pickle


app = Flask(__name__)

pre_pipeline = Pipeline(lemmatize)
pre_pipeline.add_before(preprocessing)
TEXT = Field(
    sequential=True,
    tokenize=word_tokenize,
    lower=True,
    stop_words=STOPWORDS,
    preprocessing=pre_pipeline,
)
LABELS = ["neu", "neg", "pos"]
VOCAB = {}
with open("vocab.pkl", "rb") as f:
    VOCAB = pickle.load(f)

best_config = {
    "hidden_size": 302,
    "lr": 0.00010769630091763721,
    "l2": 2.5888680371842294e-05,
    "nonlin": "tanh",
    "dropout": 0.1,
    "num_layers": 2,
    "mode": 0,
    "optimizer": "Adam",
    "momentum": 0.1,
}

best_model = SeqModel(
    embedding_size=100,
    vocab_size=len(VOCAB),
    output_size=3,
    hidden_size=best_config["hidden_size"],
    num_layers=best_config["num_layers"],
    nonlin=best_config["nonlin"],
    dropout_rate=best_config["dropout"],
    mode=best_config["mode"],
    unit="gru",
    more_features=False,
)
best_model.load_state_dict(torch.load("model_deploy.pt"))


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        tweet = request.form["search"]
        return predict_sentiment(best_model, {"tweet": tweet})[0]
    else:
        return render_template("home.html")


def preprocess(tweet):
    return [VOCAB.get(token, 0) for token in TEXT.preprocess(tweet)]


def predict_sentiment(model, input_json):
    tweet = input_json["tweet"]
    num_input = preprocess(tweet)
    model_outputs = best_model(torch.LongTensor(num_input).reshape((-1, 1)))
    probabilities, predicted = torch.max(model_outputs.cpu().data, 1)
    pred_labels = LABELS[predicted]
    return pred_labels, probabilities


if __name__ == "__main__":
    app.run(debug=True)