import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model
MODEL_PATH = "./audio_classifier.h5"  # Ensure this matches your deployed model file name
model = load_model(MODEL_PATH)

# Constants for audio processing
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109  # Ensure compatibility with training

def process_audio(file_path):
    """Preprocess the audio file to extract Mel spectrogram."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]
    
    return mel_spectrogram

# Streamlit UI
st.title("Audio Classification App")
st.write("Upload a `.flac` file to classify as bonafide or spoof.")

uploaded_file = st.file_uploader("Choose a .flac file", type=["flac"])

if uploaded_file is not None:
    with open("temp_audio.flac", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the uploaded file
    mel_spec = process_audio("temp_audio.flac")
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)   # Add batch dimension

    # Predict with the model
    prediction = model.predict(mel_spec)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = {0: "spoof", 1: "bonafide"}

    st.write("Prediction:")
    st.write(f"Class: {class_labels[predicted_class[0]]}")
    st.write(f"Confidence: {prediction[0][predicted_class[0]]:.2f}")
