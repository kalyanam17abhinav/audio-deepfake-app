import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("./audio_classifier.h5")

# Function to preprocess the .flac audio file
def preprocess_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=22050)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    # Aggregate features (mean of each MFCC coefficient across frames)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# Streamlit UI
st.title("Audio Deepfake Detection")
st.write("Upload a `.flac` file for deepfake analysis.")

# File uploader
uploaded_file = st.file_uploader("Upload your `.flac` audio file", type=["flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/flac")
    
    # Save the uploaded file temporarily
    with open("temp.flac", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess the audio file
    audio_features = preprocess_audio("temp.flac")
    
    # Reshape for model input (add batch dimension)
    audio_features = np.expand_dims(audio_features, axis=0)
    
    # Predict using the model
    prediction = model.predict(audio_features)
    result = "Deepfake Detected" if prediction[0][0] > 0.5 else "Audio is Real"
    
    # Display the result
    st.subheader("Prediction:")
    st.write(result)
