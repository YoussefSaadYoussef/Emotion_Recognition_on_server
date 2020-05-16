from flask import Flask, request, jsonify
from utilis import *
import numpy as np
import json

application = Flask(__name__)


@application.route("/")
def index():
    data = {}
    data["message"] = "Hello, your app is working!"
    return jsonify(data)

@application.route("/photo", methods=["POST"])
def index2():
    read = request.get_json()
    if type(read) == str:
        read = json.loads(read)
    img = read['img']
    img = decode(img)

    new_img, emotion = emotion_finder(img)
    sound_bytes = read_emotion(emotion)
    data = {}
    data['img'] = new_img.tolist()
    data['sound'] = str(sound_bytes)
    data['emotions'] = emotion

    return jsonify(data)
