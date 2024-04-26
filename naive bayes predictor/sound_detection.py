# sound_detection.py
import pyaudio
import numpy as np
import cv2

# Initialize PyAudio
py_audio = pyaudio.PyAudio()

# Set the frame size and hop length (in samples)
frame_rate = 44100
frame_length = int(0.04 * frame_rate)  # Frame size in samples (40ms)
hop_length = int(0.04 * frame_rate)    # Hop length in samples (40ms, non-overlapping)
window_size = int(1 * frame_rate)      # Window size in samples (1 second)

# Set the threshold value in decibels
threshold_db = 38  # Adjust this value to suit your needs

# Open an audio stream for input from the microphone
audio_stream = py_audio.open(format=pyaudio.paInt16, channels=1, rate=frame_rate, input=True, frames_per_buffer=window_size)

def calculate_db(data):
    # Calculate the RMS amplitude
    rms = np.sqrt(np.mean(data ** 2))

    # Calculate the dB level
    db = 20 * np.log10(rms)

    return db

def detect_sound(cap):
    # Read audio data from the microphone
    audio_data = np.frombuffer(audio_stream.read(window_size, exception_on_overflow=False), dtype=np.int16)

    # Sliding window over the audio data
    for i in range(0, len(audio_data) - frame_length + 1, hop_length):
        frame = audio_data[i:i + frame_length]

        # Calculate the dB level for the audio frame
        db = calculate_db(frame)

        # Determine if there is sound or not
        if db > threshold_db:
            audio_cheat = 1  # Sound detected
            break
    else:
        audio_cheat = 0  # No sound detected

    # Calculate the timestamp based on the current video frame
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get the current timestamp in milliseconds

    return timestamp, audio_cheat