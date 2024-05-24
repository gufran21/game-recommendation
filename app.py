import sys
import io
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding, chatbot
import os
from dotenv import load_dotenv
import markdown

# Set the standard output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import markdown

app = Flask(__name__)


@app.route("/")

def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    chat_history=[]
    msg = request.form["msg"]
    input = msg
    print(input)
    result,chat_history=chatbot(input,chat_history)
    result= markdown.markdown(result)
    print('history:',chat_history)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run()
