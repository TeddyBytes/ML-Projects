import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
from IPython.display import display
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

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
print(type(df))

# --------------------------------------------------------------
# Imported functions
# --------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, method, reset_index=True, dpi=300):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting (default: True)
        dpi (int): Dots Per Inch for saving the plot (default: 300)
        format (str): File format for saving the plot (e.g., 'png', 'svg', 'pdf')
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    plt.savefig(f"../../reports/figures/{col}_outliers_via_{method}.png", dpi=dpi)
    plt.show()

    return None


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        columns (list or string): The column(s) you want to apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    # display(dataset)
    # print(columns)
    dataset = dataset.copy()
    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


def plot_outlier_data(dataset, col, outlier_col, method_label, ax):
    """Plots outliers in a subplot.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        method_label (string): Label for the outlier detection method
        ax (matplotlib.axes._axes.Axes): The subplot to use for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col]).reset_index()
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
        label="no outlier " + col,
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
        label="outlier " + col,
    )

    # Set labels and legend
    ax.set_xlabel("samples")
    ax.set_ylabel("value")
    ax.legend(loc="upper center", ncol=2, fancybox=True, shadow=True)
    ax.set_title(f"{method_label} Outliers")

    return None


# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------


# Create a single figure
fig, axs = plt.subplots(nrows=1, ncols=len(exercise_names.keys()), figsize=(25, 8))

# Iterate over every exercise
for ax, exercise in zip(axs.flat, exercise_names.keys()):
    sns.boxplot(data=df.query(f"exercise == '{exercise}'"), ax=ax)
    ax.set_xlabel(exercise_names[exercise])

# Add common y-label on the left
fig.text(0.04, 0.5, "Sensor Measurement", va="center", rotation="vertical")

# Add common x-label at the bottom
fig.text(0.5, 0.04, "Exercise", ha="center")

# Add a supertitle outside the loop
fig.suptitle("Box Plot data for Sensors across all exercises")

plt.savefig(f"../../reports/figures/Box Plot displaying outliers", dpi=300)
plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Plot sensor data, notating outliers via IQR
outlier_via_iqr = df.copy()

# Iterate over every sensor
for sensor in sensor_names.keys():
    # Notate outliers via boolean
    outlier_via_iqr = mark_outliers_iqr(outlier_via_iqr, sensor)

# Iterate over every sensor
for sensor in sensor_names.keys():

    outlier_col = f"{sensor}_outlier"
    plot_binary_outliers(outlier_via_iqr, sensor, outlier_col, "iqr")

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Insert Chauvenet's function
outlier_via_chev = df.copy()

# Iterate over every sensor
for sensor in sensor_names.keys():
    outlier_via_chev = mark_outliers_chauvenet(outlier_via_chev, sensor)


# Plot sensor data, notating outliers via Chauvenets criteria
for sensor in sensor_names.keys():

    outlier_col = f"{sensor}_outlier"
    plot_binary_outliers(outlier_via_chev, sensor, outlier_col, "Chauvenets")

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

outlier_via_lof = df.copy()

# outlier_via_lof =

# Loop over all columns
outlier_via_lof, y, z = mark_outliers_lof(outlier_via_lof, sensor_names.keys())

# Plot sensor data, notating outliers via Chauvenets criteria
for sensor in sensor_names.keys():
    outlier_col = f"outlier_lof"
    plot_binary_outliers(outlier_via_lof, sensor, outlier_col, "lof")


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

dataset = df.copy()

# Create a single figure
fig, axs = plt.subplots(nrows=len(sensor_names), ncols=3, figsize=(45, 35), dpi=300)

outlier_methods = [
    (mark_outliers_iqr, "IQR", "_outlier"),
    (mark_outliers_chauvenet, "Chev", "_outlier"),
    (mark_outliers_lof, "lof", "outlier_lof"),
]

outlier_data_lof, y, z = mark_outliers_lof(dataset.copy(), sensor_names.keys())

# Iterate over every sensor
for i, sensor in enumerate(sensor_names.keys()):
    # Iterate and plot each method of outlier detection
    for j, (outlier_func, method_label, outlier_col_name) in enumerate(outlier_methods):
        if method_label == "lof":
            # Plot outliers
            plot_outlier_data(
                outlier_data_lof, sensor, "outlier_lof", method_label, axs[i][j]
            )
        else:
            # Notate outliers via boolean
            outlier_data = outlier_func(dataset.copy(), sensor)
            outlier_col = f"{sensor}{outlier_col_name}"
            # Plot outliers
            plot_outlier_data(
                outlier_data, sensor, outlier_col, method_label, axs[i][j]
            )

plt.savefig(f"../../reports/figures/Combined Outlier Data for Comparison")
# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

constrained_df = df.copy()

# Iterate over every sensor
for sensor in sensor_names.keys():
    # Notate outliers via boolean
    outlier_via_chev = mark_outliers_chauvenet(constrained_df, sensor)
    outlier_col = f"{sensor}_outlier"
    # Update values to NaN where outlier_col is True
    constrained_df.loc[outlier_via_chev[outlier_col], sensor] = np.nan


constrained_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

# Save dataframe to Interim data folder.
constrained_df.to_pickle("../../data/interim/02_outliers_removed_df.pkl")
