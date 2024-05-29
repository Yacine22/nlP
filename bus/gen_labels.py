# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:19:25 2024

@author: my220
"""

import os
import glob
import librosa
import numpy as np
from scipy.io.wavfile import write

# Directory containing the audio files
audio_dir = 'data/audio'
# Directory for augmented audio files
augmented_dir = 'data/augmented_audio'
# Output file for labels
output_file = 'data/labels.txt'

# Ensure augmented audio directory exists
os.makedirs(augmented_dir, exist_ok=True)

# Function to extract label from file name
def extract_label(file_name):
    # Remove the file extension
    label = os.path.splitext(file_name)[0]
    # Replace underscores with spaces (if any)
    label = label.replace('_', ' ')
    return label

# Function to add noise
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

# Function to shift time
def shift_time(audio, shift_max=0.2):
    shift = int(len(audio) * shift_max)
    return np.roll(audio, shift)

# Function to change pitch
def change_pitch(audio, sr=22050, pitch_factor=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

# Function to change amplitude
def change_amplitude(audio, amplitude_factor=1.5):
    return audio * amplitude_factor

# Function to convert float audio data to int16
def float_to_int16(audio):
    audio = audio / np.max(np.abs(audio))  # Normalize to -1 to 1 range
    audio_int16 = (audio * 32767).astype(np.int16)  # Scale to int16 range
    return audio_int16

# List to store the labels
labels = []

# Iterate through the audio directory
for file_name in os.listdir(audio_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(audio_dir, file_name)
        audio, sr = librosa.load(file_path, sr=None)

        # Original
        augmented_file_path = os.path.join(augmented_dir, file_name)
        write(augmented_file_path, sr, float_to_int16(audio))
        label = extract_label(file_name)
        labels.append((file_name, label))

        # Noise
        audio_noisy = add_noise(audio)
        noisy_file_name = f"noisy_{file_name}"
        noisy_file_path = os.path.join(augmented_dir, noisy_file_name)
        write(noisy_file_path, sr, float_to_int16(audio_noisy))
        labels.append((noisy_file_name, label))

        # Shift time
        audio_shifted = shift_time(audio)
        shifted_file_name = f"shifted_{file_name}"
        shifted_file_path = os.path.join(augmented_dir, shifted_file_name)
        write(shifted_file_path, sr, float_to_int16(audio_shifted))
        labels.append((shifted_file_name, label))

        # Pitch
        audio_pitched = change_pitch(audio, sr)
        pitched_file_name = f"pitched_{file_name}"
        pitched_file_path = os.path.join(augmented_dir, pitched_file_name)
        write(pitched_file_path, sr, float_to_int16(audio_pitched))
        labels.append((pitched_file_name, label))

        # Amplitude
        audio_amplified = change_amplitude(audio)
        amplified_file_name = f"amplified_{file_name}"
        amplified_file_path = os.path.join(augmented_dir, amplified_file_name)
        write(amplified_file_path, sr, float_to_int16(audio_amplified))
        labels.append((amplified_file_name, label))

# Write the labels to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    for file_name, label in labels:
        f.write(f"{file_name}\t{label}\n")

print(f"Labels have been written to {output_file}")
