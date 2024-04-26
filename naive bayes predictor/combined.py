# combined.py
import cv2
import pandas as pd
from emo import detect_emotions
from headpose import estimate_head_pose
from sound_detection import detect_sound
import pickle  # Import the pickle module to load the trained model
from sklearn.preprocessing import LabelEncoder

# Load the trained decision tree classifier
with open('gaussian_nb_model.pkl', 'rb') as f:
    nb = pickle.load(f)

# Initialize the DataFrame
df = pd.DataFrame(columns=['Timestamp', 'X_AXIS_CHEAT', 'Y_AXIS_CHEAT', 'Audio_Cheat','Emotion'])

# Initialize the plot


# Initialize an empty list to store dB values


# Open the webcam
cap = cv2.VideoCapture(0)

keep_running = True

while keep_running:
    success, image = cap.read()
    if not success:
        print("Failed to capture frame from the webcam.")
        break

    # Flip the image horizontally for a selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Estimate head pose
    processed_image, X_AXIS_CHEAT, Y_AXIS_CHEAT = estimate_head_pose(image)

    # Detect facial emotions
    emotions = detect_emotions(processed_image)

    # Detect audio from the microphone
    timestamp, Audio_Cheat = detect_sound(cap)

    # Append the data to the DataFrame
    for emotion in emotions:
        max_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
        new_row = {'Timestamp': timestamp,
                   'X_AXIS_CHEAT': X_AXIS_CHEAT,
                   'Y_AXIS_CHEAT': Y_AXIS_CHEAT,
                   'Audio_Cheat': Audio_Cheat,
                   'Emotion': max_emotion
                   }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('head_pose_emotion_detection.csv', index=False)

        # Use the decision tree classifier to predict cheating
        le = LabelEncoder()
        X = df
        print(X)
        X = X.drop(['Timestamp'], axis=1)
        X['Emotion'] = le.fit_transform(X['Emotion'])
        cheating_prediction = nb.predict(X)
        print(f"CHEATING : {cheating_prediction}")
        print("\n")

    # Convert the image back to BGR for OpenCV
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Head Pose and Emotion Detection', processed_image)

    # Check for the escape key press
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Escape key
        keep_running = False

# Clean up resources
cap.release()

cv2.destroyAllWindows()

# Save the DataFrame to a CSV file
df.to_csv('head_pose_emotion_detection.csv', index=False)


