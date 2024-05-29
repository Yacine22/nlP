import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

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
    file_path = os.path.join('data/audio', row['file'])
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
    file_path = os.path.join('data/audio', row['file'])
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
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_encoded, test_size=0.25, random_state=42)

# Reshape data for Conv1D input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the model using Conv1D
model = Sequential([
    Conv1D(96, 3, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001)),
    MaxPooling1D(2),
    Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling1D(2),
    Bidirectional(LSTM(96, return_sequences=True, kernel_regularizer=l2(0.001))),
    Bidirectional(LSTM(224, kernel_regularizer=l2(0.001))),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('saved_model/speech_classifier_model2.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Function to segment, preprocess, and predict keywords in a new audio file
def predict_audio_file_segments(file_path, segment_length=2.0, sr=22050):
    audio = load_audio_file(file_path, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = int(np.ceil(len(audio) / segment_samples))
    predictions = []

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        features = extract_features(segment)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        predictions.append(predicted_label[0])
    
    # Aggregate predictions and return all unique labels detected
    unique_predictions = list(set(predictions))
    return unique_predictions

# Test the model with a new audio file containing multiple keywords
test_file_path = 'data/tests/test2.wav'
predicted_labels = predict_audio_file_segments(test_file_path)
print(f'Predicted labels for {test_file_path}: {predicted_labels}')
