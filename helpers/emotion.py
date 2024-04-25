import cv2
from fer import FER
import pandas as pd

# Load the video
video_file = "wangyus11.avi"
video = cv2.VideoCapture(video_file)

# Create a FER detector
detector = FER(mtcnn=True)

# Create an empty list to store the data
data = []

# Get the video frame count and calculate the frame rate
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

# Loop through the video frames
for frame_idx in range(frame_count):
    # Read a frame from the video
    ret, frame = video.read()

    # If there are no more frames, break the loop
    if not ret:
        break

    # Detect emotions from the frame
    emotions = detector.detect_emotions(frame)

    # Loop through each detected face
    for face in emotions:
        # Get the emotion scores
        scores = face['emotions']
        print(face['emotions'])
        print("\n")

        # Find the emotion with the highest score
        max_emotion = max(scores, key=scores.get)

        # Calculate the time step
        time_step = frame_idx / fps

        # Append the emotion and time step to the data list
        data.append({'Emotion': max_emotion, 'Time Step': time_step})
        cv2.putText(frame, str(max_emotion), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    cv2.imshow('Emotion',frame)

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('emotions.csv', index=False)

# Release the video capture object
video.release()