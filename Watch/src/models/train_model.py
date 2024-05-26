import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df = pd.read_pickle(f"../../data/interim/04_FeatureEng_DF.pkl")

# --------------------------------------------------------------
# Define Global Variables for friendly plotting
# --------------------------------------------------------------

exercise_names = {
    "bench": "Bench Press",
    "ohp": "Overhead Press",
    "squat": "Back Squat",
    "dead": "Deadlift",
    "row": "Barbell Row",
    "rest": "Rest between Sets",
}

sensor_names = {
    "acc_x": "X-axis Acceleration (g)",
    "acc_y": "Y-axis Acceleration (g)",
    "acc_z": "Z-axis Acceleration (g)",
    "gyr_x": "X-axis Rotation Rate (deg/s)",
    "gyr_y": "Y-axis Rotation Rate (deg/s)",
    "gyr_z": "Z-axis Rotation Rate (deg/s)",
}

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
print(set(df.columns))

sensor_features = list(sensor_names.keys())
magnitude_features = ["acc_r", "gyr_r"]
pca_features = [col for col in df.columns if "pca" in col]
freq_features = [col for col in df.columns if ("freq" in col) or ("pse" in col)]
# rolling avg/std/...
temporal_features = [col for col in df.columns if "_temp" in col]

# print(len(temporal_features))

feature_set_1 = sensor_features
feature_set_2 = list(set(feature_set_1 + magnitude_features))
feature_set_3 = list(set(feature_set_2 + pca_features))
feature_set_4 = list(set(feature_set_3 + temporal_features))
feature_set_5 = list(set(feature_set_4 + freq_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
