from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
import os
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor
import torchaudio
import torch
import numpy as np 
import librosa
import time
from datetime import datetime
from lang_trans.arabic import buckwalter



app = Flask(__name__)


class Model:
    def __init__(self , model_path , processor_path):
        print("LOADING MODEL ...\n")
        start = time.time()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cpu")
        end = time.time()
        print("MODEL LOADED after : {} second\n\n".format(end-start))


        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

    def load_audio(self,path):
        batch={}
        speech_array, sampling_rate = torchaudio.load(path)
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        return batch


    def resample(self,batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
        batch["sampling_rate"] = 16_000
        return batch

    def prepare_data(self,batch):
        batch["input_values"] = self.processor(batch["speech"],
        sampling_rate=batch["sampling_rate"]).input_values
    
        return batch


    def predict(self,audio_path):
        batch = self.load_audio(audio_path)
        batch = self.resample(batch)
        batch = self.prepare_data(batch)
    
        input_dict = self.processor(batch["input_values"][0],
        return_tensors="pt", padding=True)

        logits = self.model(input_dict.input_values.to("cpu")).logits

        pred_ids = torch.argmax(logits, dim=-1)[0]
    
        return buckwalter.untransliterate(self.processor.decode(pred_ids))

    

model_path = "/home/mohammed/Downloads/Compressed/wav2vec2-large-xlsr-arabic-demo-v4/model"
processor_path = "/home/mohammed/Downloads/Compressed/wav2vec2-large-xlsr-arabic-demo-v4/processor"
  
model = Model(model_path,processor_path)


where_to_save_files = "/home/mohammed/PycharmProjects/spereco/flaskProject1/uploads"



@app.route('/', methods=['GET', 'POST'])
def main_page():
    global where_to_save_files
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
    Prediction = model.predict(file_to_predict)
    print(file_to_predict)
    return render_template('predict.html', Prediction=Prediction)


app.run()
