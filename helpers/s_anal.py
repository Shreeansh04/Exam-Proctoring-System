import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize PyAudio
py_audio = pyaudio.PyAudio()

# Open an audio stream for input from the microphone
audio_stream = py_audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Set the frame size and hop length (in samples)
frame_length = int(0.04 * 44100)  # Frame size in samples (40ms)
hop_length = int(0.04 * 44100)    # Hop length in samples (40ms, non-overlapping)

# Set the duration for capturing audio (in seconds)
duration = 20  # Adjust this value to your desired duration

# Initialize lists to store audio data and timestamps
audio_data_list = []
timestamps = []

# Start capturing audio data
print(f"Capturing audio for {duration} seconds...")
start_time = time.time()
elapsed_time = 0

while elapsed_time < duration:
    # Read audio data
    audio_frame = audio_stream.read(hop_length, exception_on_overflow=False)
    audio_data = np.frombuffer(audio_frame, dtype=np.int16)

    # Calculate the amplitude
    rms = np.sqrt(np.mean(audio_data ** 2))
    db = 20 * np.log10(rms)

    # Append the audio data and timestamp
    audio_data_list.append(rms)
    timestamps.append(elapsed_time)

    # Update elapsed time
    elapsed_time = time.time() - start_time

# Stop the audio stream
audio_stream.stop_stream()
audio_stream.close()
py_audio.terminate()

# Plot the amplitude
plt.figure(figsize=(12, 6))
plt.plot(timestamps, audio_data_list)
plt.title('Amplitude of Microphone Input')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()