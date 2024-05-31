# Standard Libraries
import time
import cProfile
import pstats

# Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn Components
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
)
from sklearn.feature_selection import SelectFromModel, RFE

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


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
# Perform one hot encoded of categorical features
# --------------------------------------------------------------

# Separate datasets into categorical/numerical features
X_train_cat = X_train.select_dtypes(exclude=[np.number])
X_test_cat = X_test.select_dtypes(exclude=[np.number])
X_train_num = X_train.select_dtypes(include=[np.number])
X_test_num = X_test.select_dtypes(include=[np.number])

# X_train_cat['participant'].value_counts()

# One-hot encode categorical features in the training and test data based on training data.
encoder = OneHotEncoder(handle_unknown="ignore")
X_train_raw_encoded = encoder.fit_transform(X_train_cat).toarray()
X_test_raw_encoded = encoder.transform(X_test_cat).toarray()

# Get column labels to encoded DataFrames
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
# Perform forward feature selection using Random Forests only
# --------------------------------------------------------------
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Initialize an empty set to store selected features
# selected_features = []

# # Create a copy of the reduced dataset
# X_train_reduced_copy = X_train_reduced.copy()

# # Train a Random Forest classifier on the full feature set
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train_reduced_copy, y_train)

# # Calculate the initial accuracy on the full feature set
# initial_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train_reduced_copy))

# # Initialize a variable to keep track of the number of iterations
# num_iterations = 0

# # Iterate until all features are selected or until no improvement in accuracy
# while len(selected_features) < X_train_reduced_copy.shape[1]:
#     # Increment the iteration counter
#     num_iterations += 1

#     # Initialize variables to keep track of the best feature to add and its corresponding accuracy
#     best_feature = None
#     best_feature_accuracy = 0

#     # Iterate over remaining features
#     for feature in X_train_reduced_copy.columns:
#         # Skip if the feature is already selected
#         if feature in selected_features:
#             continue

#         # Add the feature to the selected set
#         selected_features.append(feature)

#         # Train a Random Forest classifier on the selected features
#         rf_classifier.fit(X_train_reduced_copy[selected_features], y_train)

#         # Calculate accuracy on the selected features
#         accuracy = accuracy_score(
#             y_train, rf_classifier.predict(X_train_reduced_copy[selected_features])
#         )

#         # Update the best feature and accuracy if the current feature improves accuracy
#         if accuracy > best_feature_accuracy:
#             best_feature = feature
#             best_feature_accuracy = accuracy

#         # Remove the feature from the selected set for the next iteration
#         selected_features.remove(feature)

#     # Add the best feature to the selected set
#     selected_features.append(best_feature)

#     # Print the progress and number of iterations
#     print(
#         f"Iteration {num_iterations}: Best feature added: {best_feature}, Accuracy: {best_feature_accuracy:.4f}"
#     )

#     # Update the best accuracy
#     if best_feature is not None:
#         best_accuracy = best_feature_accuracy
#     else:
#         # Stop if no feature improves accuracy
#         break

# # Print the selected features
# print("Selected Features:", selected_features)

# selected_features_fs_rf = [
#     "set",
#     "acc_x",
#     "acc_y",
#     "acc_z",
#     "gyr_x",
#     "gyr_y",
#     "gyr_z",
#     "Duration",
#     "pca_std_acc_z",
#     "pca_std_gyr_x",
#     "pca_std_gyr_y",
#     "pca_norm_acc_x",
#     "acc_r",
#     "gyr_r",
#     "gyr_x_temp_mean_ws_5",
#     "gyr_y_temp_mean_ws_5",
#     "gyr_z_temp_mean_ws_5",
#     "gyr_r_temp_mean_ws_5",
#     "acc_z_freq_1.429_Hz_ws_14",
# ]
# --------------------------------------------------------------
# Perform Feature Elimination using feature importance of select models
# --------------------------------------------------------------

# Define the number of features to select (square root of total number of features)
n_features_to_select = int(np.sqrt(X_train_reduced.shape[1]))

# Initialize Random Forest, Gradient Boosting, and Logistic Regression Classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)

# Create SelectFromModel object for Random Forest
sfm_rf = SelectFromModel(estimator=rf_classifier, threshold="mean")
sfm_rf.fit(X_train_reduced, y_train)
selected_features_rf = X_train_reduced.columns[sfm_rf.get_support()]

# Create SelectFromModel object for Gradient Boosting
# This took more time than RF
sfm_gb = SelectFromModel(estimator=gb_classifier, threshold="mean")
sfm_gb.fit(X_train_reduced, y_train)
selected_features_gb = X_train_reduced.columns[sfm_gb.get_support()]

# Create SelectFromModel object for Logistic Regression
sfm_lr = SelectFromModel(estimator=lr_classifier, threshold="mean")
sfm_lr.fit(X_train_reduced, y_train)
selected_features_lr = X_train_reduced.columns[sfm_lr.get_support()]

# Union over all features, all features selected by at least one model.
optimal_features = (
    set(selected_features_rf) | set(selected_features_gb) | set(selected_features_lr)
)
len(optimal_features)
# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# print(set(df.columns))

pot_magnitude_features = ["acc_r", "gyr_r"]

# Extract sensor features that exist in the dataset
sensor_features = [
    sensor for sensor in list(sensor_names.keys()) if sensor in X_train_reduced.columns
]

# Extract magnitude features that exist in the dataset
magnitude_features = [
    sensor for sensor in pot_magnitude_features if sensor in X_train_reduced.columns
]

# Extract PCA features from the dataset
pca_features = [col for col in X_train_reduced.columns if "pca" in col]

# Extract frequency features from the dataset, including both "freq" and "pse" features
freq_features = [
    col for col in X_train_reduced.columns if ("freq" in col) or ("pse" in col)
]
# rolling avg/std/...
temporal_features = [col for col in X_train_reduced.columns if "_temp" in col]


# Define different feature sets
feature_set_1 = sensor_features
feature_set_2 = list(set(feature_set_1 + magnitude_features))
feature_set_3 = list(set(feature_set_2 + pca_features))
feature_set_4 = list(set(feature_set_3 + temporal_features))
feature_set_5 = list(set(feature_set_4 + freq_features))
feature_set_6 = list(selected_features_rf)
feature_set_7 = list(selected_features_gb)
feature_set_8 = list(selected_features_lr)
feature_set_10 = list(optimal_features)


# --------------------------------------------------------------
# Define parameters for Grid search
# --------------------------------------------------------------

# Define feature sets
feature_sets = {
    "Sensor Features": feature_set_1,
    "Sensor + Magnitude Features": feature_set_2,
    "Sensor + Magnitude + PCA Features": feature_set_3,
    "Sensor + Magnitude + PCA + Temporal Features": feature_set_4,
    "Sensor + Magnitude + PCA + Temporal + Frequency Features": feature_set_5,
    "Selected Features from Random Forest": feature_set_6,
    "Selected Features from Gradient Boosting": feature_set_7,
    "Selected Features from Logistic Regression": feature_set_8,
    "Optimal Features": feature_set_10,
}

# # Define models
# models = {
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "Logistic Regression": LogisticRegression(),
#     "Naive Bayes": GaussianNB(),
# }

# models = {
#     # "Random Forest": RandomForestClassifier(),
#     "Logistic Regression": LogisticRegression(),
#     "Naive Bayes": GaussianNB(),
# }
# # Define parameter grids for each model
# param_grids = {
#     "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
#     "Gradient Boosting": {
#         "n_estimators": [10, 50, 100],
#         "max_depth": [3, 5, 10],
#         "learning_rate": [0.1, 0.01, 0.001],
#     },
#     "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
#     "Naive Bayes": {}
# }

# Define parameter grids for each model
# param_grids = {
#     # "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
#     "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
#     "Naive Bayes": {},
# }

# # Initialize empty DataFrame to store results
# results_df = pd.DataFrame(
#     columns=["Model", "Feature Set", "Best Params", "Best Score", "Test Accuracy"]
# )

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

# # Split data into training and validation sets
# X_train_split, X_val, y_train_split, y_val = train_test_split(
#     X_train_reduced, y_train, test_size=0.2, random_state=42
# )
# Define models
models = {
    "Logistic Regression": Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression())]
    ),
    "Random Forest": RandomForestClassifier(),  # No standardization for Random Forest
    "Naive Bayes": GaussianNB(),
}

# Define parameter grids for each model
param_grids = {
    "Logistic Regression": {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear", "lbfgs"],
    },
    "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
    "Naive Bayes": {},
}

# Define the number of splits for Time Series Cross-Validation
n_splits = 5

# Initialize empty DataFrame to store results
results_df = pd.DataFrame(
    columns=["Model", "Feature Set", "Regular CV Score", "Time Series CV Score"]
)

# Initialize variables to track the best model and its performance
best_model = None
best_feature_set = None
best_time_series_cv_score = -1

# List to store all results
results_list = []

# Perform grid search for each model and feature set
for model_name, model in models.items():
    for feature_set_name, feature_set in feature_sets.items():
        # Regular Cross-Validation
        regular_cv_scores = cross_val_score(
            model,
            X_train_reduced[feature_set],
            y_train,
            cv=5,  # You can adjust the number of folds as needed
            scoring="accuracy",
        )
        regular_cv_score = np.mean(regular_cv_scores)

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        time_series_cv_scores = []
        for train_index, val_index in tscv.split(X_train_reduced[feature_set]):
            X_train_fold, X_val_fold = (
                X_train_reduced[feature_set].iloc[train_index],
                X_train_reduced[feature_set].iloc[val_index],
            )
            y_train_fold, y_val_fold = (
                y_train.iloc[train_index],
                y_train.iloc[val_index],
            )
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=5, scoring="accuracy"
            )
            grid_search.fit(X_train_fold, y_train_fold)
            time_series_cv_scores.append(grid_search.best_score_)
        time_series_cv_score = np.mean(time_series_cv_scores)

        # Update best model based on Time Series CV score
        if time_series_cv_score > best_time_series_cv_score:
            best_time_series_cv_score = time_series_cv_score
            best_model = model_name
            best_feature_set = feature_set_name
            best_params = grid_search.best_params_

        # Store results
        results_list.append(
            {
                "Model": model_name,
                "Feature Set": feature_set_name,
                "Regular CV Score": regular_cv_score,
                "Time Series CV Score": time_series_cv_score,
            }
        )

        # Print progress
        print(
            f"Model: {model_name}, Feature Set: {feature_set_name}, Regular CV Score: {regular_cv_score}, Time Series CV Score: {time_series_cv_score}"
        )

# Convert results list into a DataFrame
results_df = pd.DataFrame(results_list)

# Sort results by Time Series CV score
results_df_sorted = results_df.sort_values(by="Time Series CV Score", ascending=False)

# Display best model and its details
print(f"Best Model: {best_model}")
print(f"Feature Set: {best_feature_set}")
print(f"Best Params: {best_params}")
print(f"Best Time Series Cross-Validation Score: {best_time_series_cv_score}")


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

# # Evaluate the best model on the test set
# X_test_best_feature_set = X_test_reduced[feature_sets[best_feature_set]]
# y_test_pred = best_model.predict(X_test_best_feature_set)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Test Set Accuracy of the selected model: {test_accuracy}")


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
