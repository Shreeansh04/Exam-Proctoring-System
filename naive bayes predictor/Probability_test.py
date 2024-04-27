#P(cheat | features) = ( P(features | cheat)* P(cheat)) / P(features) - for testing/real time
import pandas as pd
from collections import Counter

# Load the dataset
data = pd.read_csv('G:/AI_end/Exam-Proctoring-System/combined datasets/head_pose_emotion_detection.csv')

# Separate the features and target variable
features = data[['X_AXIS_CHEAT', 'Y_AXIS_CHEAT', 'Audio_Cheat']]
target = data['Cheating']

# Calculate the prior probability P(cheat)
total_instances = len(data)
cheat_instances = sum(target == 1)
p_cheat = cheat_instances / total_instances #fine if you say so

# Given P(features|cheat) values
p_features_given_cheat = {
    (1, 0, 1): 1.0000,
    (1, 1, 0): 1.0000,
    (0, 1, 0): 0.9999,
    (0, 0, 0): 0.9999,
    (1, 0, 0): 1.0000,
    (0, 0, 1): 1.0000,
    (1, 1, 1): 1.0000,
    (0, 1, 1): 1.0000
}

# Given P(features|~cheat) values
p_features_given_non_cheat = {
    (1, 0, 1): 0.0000,
    (1, 1, 0): 0.0000,
    (0, 1, 0): 0.0001,
    (0, 0, 0): 0.0001,
    (1, 0, 0): 0.0000,,
    (0, 0, 1): 0.0000,
    (1, 1, 1): 0.0000,
    (0, 1, 1): 0.0000
}

# Calculate P(cheat|features)
for feature_combination in set(p_features_given_cheat.keys()):
    p_features_given_cheat_val = p_features_given_cheat[feature_combination]
    p_features_given_non_cheat_val = p_features_given_non_cheat[feature_combination]
    p_features = p_features_given_cheat_val * p_cheat + p_features_given_non_cheat_val * (1 - p_cheat)

    if p_features > 0:
        p_cheat_given_features = (p_features_given_cheat_val * p_cheat) / p_features
        print(f"P(cheat | {feature_combination}) = {p_cheat_given_features:.4f}")
    else:
        print(f"P(cheat | {feature_combination}) = 0 (no instances of this feature combination in the dataset)")