from flask import Flask, request, redirect, url_for, render_template
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from werkzeug.utils import secure_filename
import librosa
import torch
import os

app = Flask(__name__)

where_to_save_files = "/home/aymenha2021/PycharmProjects/flaskProject1/uploads"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    "/home/aymenha2021/PycharmProjects/flaskProject/wav2vec2-large-xlsr-arabic")
model = Wav2Vec2ForCTC.from_pretrained("/home/aymenha2021/PycharmProjects/flaskProject/wav2vec2-large-xlsr-arabic")


def prepare_example(example):
    speech, sampling_rate = librosa.load(example)
    return speech


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if not os.path.exists(where_to_save_files):
            os.makedirs(where_to_save_files)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    file_to_predict = os.path.join('uploads', filename)
    file_to_predict = prepare_example(file_to_predict)
    inputs = tokenizer(file_to_predict, return_tensors="pt").input_values
    logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    Prediction = tokenizer.batch_decode(predicted_ids)

    return render_template('predict.html', Prediction=Prediction)


app.run()
