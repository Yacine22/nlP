# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:23:58 2024

@author: my220
"""

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pathlib

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('saved_model/speech_classifier_model.h5')

# Load label encoder
labels_df = pd.read_csv('data/labels.txt', sep='\t', header=None, names=['file', 'label'], encoding='utf-8')
label_encoder = LabelEncoder()
label_encoder.fit(labels_df['label'])

# Function to load and preprocess audio files
def load_audio_file(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Extract MFCC features
def extract_features(audio, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join('data/tests', file.filename)
        print("Loading wav file")
        #file.save(file_path)
        
        audio = load_audio_file(file_path)
        features = extract_features(audio)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        
        return render_template('result.html', label=predicted_label[0], file=pathlib.Path(file_path).name)

if __name__ == '__main__':
    app.run(debug=True)
