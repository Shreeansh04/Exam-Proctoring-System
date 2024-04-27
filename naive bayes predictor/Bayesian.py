#P(cheat | features) = (P(features | cheat) * P(cheat)) / P(features)
import numpy as np
import pandas as pd
from collections import Counter

# Load the dataset
data = pd.read_csv('C:\\college\\ai_assg\\Exam-Proctoring-System\\combined datasets\\training set.csv')

# Separate the features and target variable
features = data[['X_AXIS_CHEAT', 'Y_AXIS_CHEAT', 'Audio_Cheat']]
target = data['Cheating']

# Calculate the prior probability P(cheat)
total_instances = len(data)
cheat_instances = sum(target == 1)
p_cheat = cheat_instances / total_instances

# Calculate P(features | cheat) and P(features | ~cheat)
feature_combinations = list(features.itertuples(index=False, name=None))
p_features_given_cheat = Counter()
p_features_given_non_cheat = Counter()

for feature_values, cheat in zip(feature_combinations, target):
    if cheat == 1:
        p_features_given_cheat[feature_values] += 1
    else:
        p_features_given_non_cheat[feature_values] += 1

p_features_given_cheat = {k: v / cheat_instances for k, v in p_features_given_cheat.items()}
p_features_given_non_cheat = {k: v / (total_instances - cheat_instances) for k, v in p_features_given_non_cheat.items()}

# Calculate P(cheat | features)
for feature_combination in set(feature_combinations):
    p_features_given_cheat_val = p_features_given_cheat.get(feature_combination, 0)
    p_features_given_non_cheat_val = p_features_given_non_cheat.get(feature_combination, 0)
    p_features = p_features_given_cheat_val * p_cheat + p_features_given_non_cheat_val * (1 - p_cheat)

    if p_features > 0:
        p_cheat_given_features = (p_features_given_cheat_val * p_cheat) / p_features
        print(f"P(cheat | {feature_combination}) = {p_cheat_given_features:.4f}")
    else:
        print(f"P(cheat | {feature_combination}) = 0 (no instances of this feature combination in the dataset)")




