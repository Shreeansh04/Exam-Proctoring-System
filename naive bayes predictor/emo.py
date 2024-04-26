# emotion.py
from fer import FER

detector = FER(mtcnn=True)

def detect_emotions(frame):
    emotions = detector.detect_emotions(frame)
    return emotions