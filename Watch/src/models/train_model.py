import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

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
df = pd.read_pickle("../../data/interim/05_FeatureEng_NO_NULL_DF.pkl")
df["Duration"] = df["Duration"].astype(int)
# df['Duration'].dtype


X, y = df.drop("exercise", axis=1), df["exercise"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


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
# print(set(df.columns))

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
# Perform one hot encoded of categorical features
# --------------------------------------------------------------

# Separate datasets into categorical/numerical features
X_train_cat = X_train.select_dtypes(exclude=[np.number])
X_test_cat = X_test.select_dtypes(exclude=[np.number])
X_train_num = X_train.select_dtypes(include=[np.number])
X_test_num = X_test.select_dtypes(include=[np.number])

# One-hot encode categorical features in the training and test data based on training data.
encoder = OneHotEncoder(
    handle_unknown="ignore", sparse=False
)  # Adjust parameters as needed
X_train_raw_encoded = encoder.fit_transform(X_train_cat)
X_test_raw_encoded = encoder.transform(X_test_cat)

# Get column labels to encoded DataFrames
categorical_columns = list(X_train_cat.columns)
encoded_column_names = encoder.get_feature_names_out(X_train_cat.columns)

# Convert one-hot encoded data back to DataFrame
X_train_encoded_df = pd.DataFrame(
    X_train_raw_encoded, index=X_train_cat.index, columns=encoded_column_names
)
X_test_encoded_df = pd.DataFrame(
    X_test_raw_encoded, index=X_test_cat.index, columns=encoded_column_names
)

# Join encoded DataFrames with numerical versions
X_train_encoded = pd.concat([X_train_num, X_train_encoded_df], axis=1)
X_test_encoded = pd.concat([X_test_num, X_test_encoded_df], axis=1)

# --------------------------------------------------------------
# Perform Correlation
# --------------------------------------------------------------
# X_train['Duration'].dtype

X_train_numeric = X_train_encoded.select_dtypes(include=[np.number])
# df_numeric['Duration']
X_train_numeric.corr()


# Calculate the correlation matrix
correlation_matrix = X_train_numeric.corr()

# Define a threshold for identifying highly correlated features
threshold = 0.8

# Create a mask to identify highly correlated features
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Find features with correlation greater than the threshold
high_correlation_features = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > threshold)
]

# Store number of highly correlated features to be removed.
num_corr_feats = len(high_correlation_features)

print(f"Number of highly correlated features is: {num_corr_feats}")


# Removed Highly Correlated features
X_train_reduced = X_train_encoded.drop(columns=high_correlation_features)
X_test_reduced = X_test_encoded.drop(columns=high_correlation_features)

# Ensure no null values
# X_train_reduced.isna().any()[X_train_reduced.isna().any()]
# X_train_reduced['acc_r_freq_1.429_Hz_ws_14'].isna().sum()

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Perform Recursive Feature Elimination using simple random forest feature importance
# --------------------------------------------------------------

# Define the number of features to select (square root of total number of features)
n_features_to_select = int(np.sqrt(X_train_reduced.shape[1]))

# Initialize Random Forest Classifier and SVC
rf_classifer = RandomForestClassifier()
sv_classifier = SVC(kernel="linear")

# Initialize RFE for both models
rf_rfe = RFE(estimator=rf_classifer, n_features_to_select=n_features_to_select)
svc_rfe = RFE(estimator=sv_classifier, n_features_to_select=n_features_to_select)

# Fit RFE to both models
rf_rfe.fit(X_train_reduced, y_train)
svc_rfe.fit(X_train_reduced, y_train)

rf_cols = X_train_reduced.columns[rf_rfe.support_]
svc_cols = X_train_reduced.columns[svc_rfe.support_]

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
