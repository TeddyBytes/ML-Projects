import pandas as pd
import regex as re
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


data_path = "../../data/raw/*.csv"
files = glob(data_path)
# files

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

exercise = files[0].split("-")[1]
participant = files[0].split("-")[0][-1]
difficulty = files[0].split("-")[2].rstrip("1234567890")
difficulty = difficulty.split("_")[0]

rpe_pattern = r"rpe(\d)"
match = re.search(rpe_pattern, files[0])
# Will output none if no RPE is supplied
rpe = match.group()


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

# Initialize variables
all_acc_df = pd.DataFrame()
all_gyr_df = pd.DataFrame()
acc_curr_set = 1
gyr_curr_set = 1

# Iterate over files in folder
for file in files:

    # Extract features from filename
    exercise = file.split("-")[1]
    participant = file.split("-")[0][-1]
    difficulty = file.split("-")[2].rstrip("1234567890")
    difficulty = difficulty.split("_")[0]

    # Extract RPE as rate
    if "rpe" in file:
        rpe_pattern = r"rpe(\d)"
        match = re.search(rpe_pattern, file)
        # Will output none if no RPE is supplied
        rpe = match.group()
    else:
        rpe = None

    # Read file
    df = pd.read_csv(file)

    # Add extracted features
    df["difficulty"] = difficulty
    df["exercise"] = exercise
    df["participant"] = participant

    # Build Respective DF's
    if "Accelerometer" in file:
        df["set"] = acc_curr_set
        acc_curr_set += 1

        all_acc_df = pd.concat([df, all_acc_df])
    else:
        df["set"] = gyr_curr_set
        gyr_curr_set += 1

        all_gyr_df = pd.concat([df, all_gyr_df])

    # print(exercise, participant, difficulty, rpe)
all_acc_df
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
