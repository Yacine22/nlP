# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:10:13 2024

@author: my220
"""

import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

# Load labels
labels_df = pd.read_csv('data/labels.txt', sep='\t', header=None, names=['file', 'label'], encoding='utf-8')

# Function to load and preprocess audio files
def load_audio_file(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Extract MFCC features
def extract_features(audio, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Data augmentation functions
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def shift_time(audio, shift_max=0.2):
    shift = int(len(audio) * shift_max)
    return np.roll(audio, shift)

def change_pitch(audio, sr=22050, pitch_factor=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

# Prepare dataset
X = []
y = []

for index, row in labels_df.iterrows():
    file_path = os.path.join('data/augmented_audio', row['file'])
    audio = load_audio_file(file_path)
    features = extract_features(audio)
    X.append(features)
    y.append(row['label'])

X = np.array(X)
y = np.array(y)

# Apply augmentations and extract features
X_augmented = []
y_augmented = []

for index, row in labels_df.iterrows():
    file_path = os.path.join('data/augmented_audio', row['file'])
    audio = load_audio_file(file_path)
    
    # Original
    features = extract_features(audio)
    X_augmented.append(features)
    y_augmented.append(row['label'])
    
    # Noise
    audio_noisy = add_noise(audio)
    features_noisy = extract_features(audio_noisy)
    X_augmented.append(features_noisy)
    y_augmented.append(row['label'])
    
    # Shift time
    audio_shifted = shift_time(audio)
    features_shifted = extract_features(audio_shifted)
    X_augmented.append(features_shifted)
    y_augmented.append(row['label'])
    
    # Pitch
    audio_pitched = change_pitch(audio)
    features_pitched = extract_features(audio_pitched)
    X_augmented.append(features_pitched)
    y_augmented.append(row['label'])

X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_augmented)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_encoded, test_size=0.33, random_state=42)

# Reshape data for Conv1D input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define model building function for Keras Tuner
# Define model building function for Keras Tuner
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from kerastuner.tuners import RandomSearch
def build_model(hp):
    model = Sequential([
        Conv1D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Int('conv_1_kernel', min_value=3, max_value=5, step=1),
            activation='relu',
            input_shape=(X_train.shape[1], 1),
            kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Int('conv_2_kernel', min_value=3, max_value=5, step=1),
            activation='relu',
            kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling1D(2),
        Bidirectional(LSTM(
            units=hp.Int('lstm_1_units', min_value=64, max_value=256, step=32),
            return_sequences=True,
            kernel_regularizer=l2(0.001)
        )),
        Dropout(0.5),
        Bidirectional(LSTM(
            units=hp.Int('lstm_2_units', min_value=64, max_value=256, step=32),
            kernel_regularizer=l2(0.001)
        )),
        Dropout(0.5),
        Dense(
            units=hp.Int('dense_units', min_value=64, max_value=256, step=32),
            activation='relu',
            kernel_regularizer=l2(0.001)
        ),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Initialize Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='speech_classification'
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]
    
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    
    # Rebuild the best model
    best_model = build_model(best_hps)
    
    # Train the best model
    best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    # Evaluate the best model
    val_loss, val_acc = best_model.evaluate(X_val, y_val)
    all_scores.append(val_acc)
    print(f'Fold validation accuracy: {val_acc}')

# Calculate the average accuracy across all folds
average_acc = np.mean(all_scores)
print(f'Average cross-validation accuracy: {average_acc}')

# Save the best model
best_model.save('saved_model/speech_classifier_model.h5')

# Function to preprocess and predict a new audio file
def predict_audio_file(file_path):
    audio = load_audio_file(file_path)
    features = extract_features(audio)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = best_model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Test the model with a new audio file
test_file_path = 'data/audio/Arsanel.wav'
predicted_label = predict_audio_file(test_file_path)
print(f'Predicted label for {test_file_path}: {predicted_label}')
