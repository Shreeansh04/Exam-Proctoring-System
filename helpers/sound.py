import pyaudio
import wave
import numpy as np
import pandas as pd

# Set the audio file path
audio_file_path = "wangyus1.wav"

# Open the audio file
with wave.open(audio_file_path, 'r') as wav:
    # Get audio parameters
    frame_rate = wav.getframerate()
    num_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()

    # Set the frame size and hop length (in samples)
    frame_length = int(0.04 * frame_rate)  # Frame size in samples (40ms)
    hop_length = int(0.04 * frame_rate)    # Hop length in samples (40ms, non-overlapping)

    # Set the threshold value in decibels
    threshold_db = 42.496  # Adjust this value to suit your needs

    # Initialize the DataFrame
    df = pd.DataFrame(columns=['Timestamp', 'Audio_Cheat'])

    def calculate_db(data):
        # Calculate the RMS amplitude
        rms = np.sqrt(np.mean(data ** 2))

        # Calculate the dB level
        db = 20 * np.log10(rms)

        return db

    def detect_sound(frame_index, data):
        # Calculate the dB level for the audio frame
        db = calculate_db(data)

        # Determine if there is sound or not
        if db > threshold_db:
            audio_cheat = 1  # Sound detected
        else:
            audio_cheat = 0  # No sound detected

        # Calculate the timestamp based on the frame index
        timestamp = frame_index * (hop_length / frame_rate)

        return timestamp, audio_cheat

    def main():
        print("Starting audio cheating detection...")
        frame_index = 0
        while True:
            try:
                # Read audio data from the file
                data = wav.readframes(hop_length)
                if not data:
                    break  # End of file

                # Convert the audio data to a NumPy array
                data = np.frombuffer(data, dtype=np.int16)

                # Detect sound and get the timestamp
                timestamp, audio_cheat = detect_sound(frame_index, data)

                # Append the data to the DataFrame
                new_row = {'Timestamp': timestamp, 'Audio_Cheat': audio_cheat}
                df.loc[len(df)] = new_row

                # Print the current status
                if audio_cheat == 1:
                    print(f"[{timestamp:.2f}s] Audio detected - Possible cheating")
                else:
                    print(f"[{timestamp:.2f}s] No audio detected")

                frame_index += 1

            except KeyboardInterrupt:
                print("Stopping audio cheating detection...")
                break

        # Save the DataFrame to a CSV file
        df.to_csv('audio_subject19.csv', index=False)

    if __name__ == "__main__":
        main()