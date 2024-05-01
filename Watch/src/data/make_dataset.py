import pandas as pd
import numpy as np
import regex as re
import datetime
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

one_acc_data = pd.read_csv(
    "../../data/raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
one_gyro_data = pd.read_csv(
    "../../data/raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/
# --------------------------------------------------------------


def list_src_data(filepath):
    """_summary_

    Args:
        filepath : Filepath of source data

    Returns:
        Array: List of files in that directory
    """

    files = glob(filepath)

    return files


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------


def extract_features_from_title(file):
    """Extracts features (exercise, participant, difficulty, RPE) from a filename.

    Args:
        file (str): The filename to extract features from.

    Returns:
        dict: A dictionary containing the extracted features.
    """

    exercise = file.split("-")[1]
    participant = file.split("-")[0][-1]
    difficulty = file.split("-")[2].rstrip("1234567890")
    difficulty = difficulty.split("_")[0]

    # Extract RPE as rate
    if "rpe" in file:
        rpe_pattern = r"rpe(\d)"
        match = re.search(rpe_pattern, file)
        # Will output none if no RPE is supplied
        rpe = match.group() if match else None
    else:
        rpe = None

    return {
        "exercise": exercise,
        "participant": participant,
        "difficulty": difficulty,
        "RPE": rpe,
    }


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

# Initialize empty DataFrames to store ACC and GYR data
all_acc_df = pd.DataFrame()
all_gyr_df = pd.DataFrame()

# Keep track of current set numbers for ACC and GYR data
acc_curr_set = 1
gyr_curr_set = 1

# Iterate over each file in folder
for file in files:

    features = extract_features_from_title(file)

    # Read data from current file
    df = pd.read_csv(file)

    # Add extracted features
    df["difficulty"] = features["difficulty"]
    df["exercise"] = features["exercise"]
    df["participant"] = features["participant"]

    # Build Respective DF's
    if "Accelerometer" in file:

        df["Set"] = acc_curr_set
        acc_curr_set += 1
        all_acc_df = pd.concat([df, all_acc_df])

    elif "Gyroscope" in file:

        df["Set"] = gyr_curr_set
        gyr_curr_set += 1
        all_gyr_df = pd.concat([df, all_gyr_df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
def process_sensor_timestamps(
    acc_df,
    gyr_df,
    time_column_name="time",
    dropped_columns=["epoch (ms)", "time (01:00)", "elapsed (s)"],
):
    """
    Processes a sensor data DataFrame by converting timestamps, adding a time column,
    and dropping unnecessary columns.

    Args:
        acc_df (pandas.DataFrame): The DataFrame containing Accelerometer data.
        gyr_df (pandas.DataFrame): The DataFrame containing Gyroscope data.
        time_column_name (str, optional): The name for the new time column (default "time").
        dropped_columns (list, optional): A list of column names to drop (default).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the processed DataFrames.
            - The first element is the DataFrame for Accelerometer Data.
            - The second element is the DataFrame for Gyroscope Data.
    """
    # Convert timestamps (epoch milliseconds) to datetime format
    acc_datetime = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_datetime = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Create a new column named "time" in both DataFrames
    acc_df["time"] = acc_datetime
    gyr_df["time"] = gyr_datetime

    # Drop unnecessary columns from the DataFrames
    acc_df.drop(dropped_columns, axis=1, inplace=True)
    gyr_df.drop(dropped_columns, axis=1, inplace=True)

    return acc_df, gyr_df

    # Explanation of dropped columns:
    #   - "epoch (ms)": Original timestamp representation, not needed anymore
    #   - "time (01:00)": Duplicate time column or unnecessary format
    #   - "elapsed (s)": Redundant since time information is sufficient


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

data_path = "../../data/raw/*.csv"


def read_data_from_files(data_path):

    files = list_src_data(data_path)

    all_acc_df = pd.DataFrame()
    all_gyr_df = pd.DataFrame()

    # Keep track of current set numbers for ACC and GYR data
    acc_curr_set = 1
    gyr_curr_set = 1

    for file in files:

        features = extract_features_from_title(file)

        # Read data from current file
        df = pd.read_csv(file)

        # Add extracted features
        df["difficulty"] = features["difficulty"]
        df["exercise"] = features["exercise"]
        df["participant"] = features["participant"]

        # Build Respective DF's
        if "Accelerometer" in file:
            df["Set"] = acc_curr_set
            acc_curr_set += 1
            all_acc_df = pd.concat([df, all_acc_df])
        elif "Gyroscope" in file:
            df["Set"] = gyr_curr_set
            gyr_curr_set += 1
            all_gyr_df = pd.concat([df, all_gyr_df])
    all_acc_df, all_gyr_df = process_sensor_timestamps(all_acc_df, all_gyr_df)

    return all_acc_df, all_gyr_df


acc_df, gyr_df = read_data_from_files(data_path)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


def merge_datasets(acc_df, gyr_df):
    """
    Merges two data sets together, particularly Accelerometer and Gyroscope data.

    Args:
        acc_df (pandas.DataFrame): Accelerometer Data frame for merging.
        gyr_df (pandas.DataFrame): Gyroscope Data frame for merging.

    Returns:
        merged_df (pandas.DataFrame): Return merged dataframe, containing both Gyroscope and Accelerometer data.
    """
    merged = pd.concat([acc_df, gyr_df])
    duplicate_cols = ["difficulty", "exercise", "participant", "Set", "time"]
    unique_df = merged.drop_duplicates(subset=duplicate_cols, keep="first")

    return unique_df


final_df = merge_datasets(acc_df, gyr_df)

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
