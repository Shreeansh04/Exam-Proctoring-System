#P(features | cheat) = ( P(cheat | features)* P(features)) / P(cheat) - for training


import numpy as np
import pandas as pd
from collections import Counter

# Load the dataset
data = pd.read_csv('G:/AI_end/Exam-Proctoring-System/combined datasets/training set.csv')

# Separate the features and target variable
features = data[['X_AXIS_CHEAT', 'Y_AXIS_CHEAT', 'Audio_Cheat']]
target = data['Cheating']

# Calculate P(cheat)
total_instances = len(data)
cheat_instances = sum(target == 1)
p_cheat = cheat_instances / total_instances

# Calculate P(features)
feature_combinations = list(features.itertuples(index=False, name=None))
p_features = Counter(feature_combinations)
p_features = {k: v / total_instances for k, v in p_features.items()}

# Calculate P(cheat|features) and P(cheat|~features)
p_cheat_given_features = {}
p_cheat_given_non_features = {}

for feature_combination in set(feature_combinations):
    cheat_instances_with_features = sum((target == 1) & (features == np.array(feature_combination)).all(axis=1))
    non_cheat_instances_with_features = sum((target == 0) & (features == np.array(feature_combination)).all(axis=1))

    p_cheat_given_features[feature_combination] = cheat_instances_with_features / p_features[feature_combination]
    p_cheat_given_non_features[feature_combination] = non_cheat_instances_with_features / (total_instances - p_features[feature_combination])

# Calculate P(features|cheat)
p_features_given_cheat = {}
p_features_given_non_cheat = {}

for feature_combination in set(feature_combinations):
    p_features_given_cheat[feature_combination] = (p_cheat_given_features[feature_combination] * p_cheat) / (
        p_cheat_given_features[feature_combination] * p_cheat + p_cheat_given_non_features[feature_combination] * (1 - p_cheat))
    p_features_given_non_cheat[feature_combination] = (p_cheat_given_non_features[feature_combination] * (1 - p_cheat)) / (
        p_cheat_given_non_features[feature_combination] * (1 - p_cheat) + p_cheat_given_features[feature_combination] * p_cheat)

# Print the calculated probabilities
print("P(features|cheat):")
for feature_values, probability in p_features_given_cheat.items():
    print(f"{feature_values}: {probability:.4f}")

print("\nP(features|~cheat):")
for feature_values, probability in p_features_given_non_cheat.items():
    print(f"{feature_values}: {probability:.4f}")

#outpput:P(features|cheat):
# (1, 0, 1): 1.0000
# (1, 1, 0): 1.0000
# (0, 1, 0): 0.9999
# (0, 0, 0): 0.9999
# (1, 0, 0): 1.0000
# (0, 0, 1): 1.0000
# (1, 1, 1): 1.0000
# (0, 1, 1): 1.0000



