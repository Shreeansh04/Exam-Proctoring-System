import librosa
import numpy as np
import pandas as pd
import datetime as dt

# place holders and global variables
SOUND_AMPLITUDE = 0
AUDIO_CHEAT = 0

# sound variables
# SUS means next sound packet is worth analyzing
CALLBACKS_PER_SECOND = 38               # callbacks per sec(system dependent)
SUS_FINDING_FREQUENCY = 2               # calculates SUS *n* times every sec
SOUND_AMPLITUDE_THRESHOLD = 10          # amplitude considered for SUS calc

# packing *n* frames to calculate SUS
FRAMES_COUNT = int(CALLBACKS_PER_SECOND/SUS_FINDING_FREQUENCY) - 18
AMPLITUDE_LIST = list([0]*FRAMES_COUNT)
SUS_COUNT = 0
count = 0

def print_sound(y, sr):
    global SOUND_AMPLITUDE, SUS_COUNT, count, SOUND_AMPLITUDE_THRESHOLD, AUDIO_CHEAT
    vnorm = np.mean(np.abs(y))
    AMPLITUDE_LIST.append(vnorm)
    count += 1
    print(FRAMES_COUNT)
    AMPLITUDE_LIST.pop(0)
    if count == FRAMES_COUNT:
        SOUND_AMPLITUDE = np.mean(AMPLITUDE_LIST)
        if SUS_COUNT >= 2:
            print("!!!!!!!!!!!! FBI OPEN UP !!!!!!!!!!!!\n")
            AUDIO_CHEAT = 1
            SUS_COUNT = 0
        if SOUND_AMPLITUDE > SOUND_AMPLITUDE_THRESHOLD:
            SUS_COUNT += 1
            print("Sus...\n", SUS_COUNT)
        else:
            SUS_COUNT = 0
            AUDIO_CHEAT = 0
        count = 0
    return AUDIO_CHEAT

def sound(file_path):
    y, sr = librosa.load(file_path)
    frames = y.shape[0] // sr
    for i in range(0, y.shape[0], int(sr / CALLBACKS_PER_SECOND)):
        print_sound(y[i:i+int(sr / CALLBACKS_PER_SECOND)], sr)

def sound_analysis(file_path):
    global df
    y, sr = librosa.load(file_path)
    df = pd.DataFrame(columns=['Timestamp', 'AUDIO_CHEAT'])
    frames = y.shape[0] // sr
    for i in range(0, y.shape[0], int(sr / CALLBACKS_PER_SECOND)):
        AUDIO_CHEAT = print_sound(y[i:i+int(sr / CALLBACKS_PER_SECOND)], sr)
        # print(AUDIO_CHEAT)
        timestamp = dt.datetime.fromtimestamp(i / sr) + dt.timedelta(milliseconds=40)
        new_row = pd.DataFrame([{'Timestamp': timestamp, 'AUDIO_CHEAT': AUDIO_CHEAT}])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('audio_cheat_detection.csv', index=False)

# Load the audio file
# y, sr = librosa.load('Yousef.wav')

# Initialize the dataframe
sound_analysis('Yousef.wav')


# Loop through the audio file
# for i in range(0, y.shape[0], int(sr / CALLBACKS_PER_SECOND)):
#     # Calculate the sound amplitude
#     vnorm = np.mean(np.abs(y[i:i+int(sr / CALLBACKS_PER_SECOND)]))
#
#     # Calculate the timestamp
#     timestamp = dt.datetime.fromtimestamp(i / sr) + dt.timedelta(milliseconds=40)
#
#     # Log the sound amplitude to the dataframe
#     #df = df.append({'Timestamp': timestamp, 'AUDIO_CHEAT': AUDIO_CHEAT}, ignore_index=True)
#     new_row = pd.DataFrame([{'Timestamp': timestamp, 'AUDIO_CHEAT': AUDIO_CHEAT}])
#     df = pd.concat([df, new_row], ignore_index=True)
#
# # Save the dataframe as a CSV file
# df.to_csv('audio_cheat_detection.csv', index=False)