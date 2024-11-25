import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display

# Function to load and preprocess the audio
def preprocess_audio(audio_path, sample_rate=16000, duration=2, n_mels=128):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # Pad or truncate to fit model input size
    max_time_steps = 128  # Adjust based on model's requirements
    if log_mel_spectrogram.shape[1] < max_time_steps:
        pad_width = max_time_steps - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_time_steps]

    return log_mel_spectrogram

# Load the model
model = load_model("./audio_classifier.h5")  # Path to the saved model file

# Streamlit App UI
st.title("Audio Deepfake Detection")

# File uploader for the audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the uploaded audio file
    audio_features = preprocess_audio("temp_audio_file.wav")

    # Reshape the data to fit the model input
    audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
    audio_features = np.expand_dims(audio_features, axis=-1)  # Add channel dimension if required

    # Make predictions
    prediction = model.predict(audio_features)
    predicted_class = np.argmax(prediction, axis=-1)

    # Display results
    st.write(f"Predicted class: {predicted_class[0]}")
    
    # Visualize the Mel spectrogram of the uploaded audio file
    st.write("Mel Spectrogram of the Audio:")
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_features[0, ..., 0], sr=16000)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    st.pyplot(plt)
