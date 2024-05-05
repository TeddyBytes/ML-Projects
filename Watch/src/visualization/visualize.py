import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df.info()

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# sns.pairplot(df[df["set"]==1])
plt.plot(df[df["set"] == 1]["acc_y"].reset_index(drop=True))
plt.xlabel("Sample Time")
plt.plot(df[df["set"] == 2]["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------


exercises = df["exercise"].unique()
mpl.style.use("seaborn-v0_8-deep")


# Create the figure and subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 8))

# Loop through exercises and subplots
for i, exercise in enumerate(exercises):
    print(i, exercise)
    selected_df = df[df["exercise"] == exercise]

    # Access the current subplot using indexing
    ax = axs.flat[i]

    # Plot data on the current subplot
    ax.plot(selected_df[:100]["acc_y"].reset_index(drop=True), label=exercise)
    ax.set_ylabel("Acc Y")
    ax.set_xlabel("Sample Time")
    plt.xlabel("Sample Time")

plt.suptitle("acc_y")
plt.show()


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 8))
# Loop through exercises and subplots
for i, exercise in enumerate(exercises):
    selected_df = df[df["exercise"] == exercise]

    # Access the current subplot using indexing
    ax = axs.flat[i]

    # Plot data on the current subplot
    ax.plot(selected_df[:100]["acc_x"].reset_index(drop=True), label=exercise)
    ax.set_ylabel("Acc X")
    ax.set_xlabel("Sample Time")
    plt.xlabel("Sample Time")


plt.suptitle("acc_x")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 8))
# Loop through exercises and subplots
for i, exercise in enumerate(exercises):
    selected_df = df[df["exercise"] == exercise]

    # Access the current subplot using indexing
    ax = axs.flat[i]

    # Plot data on the current subplot
    ax.plot(selected_df[:100]["gyr_x"].reset_index(drop=True), label=exercise)
    ax.set_ylabel("gyr_x ")
    ax.set_xlabel("Sample Time")
    plt.xlabel("Sample Time")


plt.suptitle("gyr_x")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 8))
# Loop through exercises and subplots
for i, exercise in enumerate(exercises):
    selected_df = df[df["exercise"] == exercise]

    # Access the current subplot using indexing
    ax = axs.flat[i]

    # Plot data on the current subplot
    ax.plot(selected_df[:100]["gyr_y"].reset_index(drop=True), label=exercise)
    ax.set_ylabel("gyr_y")
    ax.set_xlabel("Sample Time")
    plt.xlabel("Sample Time")

# SUP not sub
fig.suptitle("gyr_y")
plt.show()

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------





# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

exercises = df["exercise"].unique()
difficulties = df["difficulty"].value_counts()
# Create the figure and subplots

sensor_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# For each sensor column
for sensor in sensor_cols:
    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 8))
    
    # Iterate over exercises and plot grouped data on each subplot
    for ax, exercise in zip(axs.flat, exercises):
        # Filter dataframe for the current exercise
        selected_df = df[df['exercise'] == exercise].sort_values('participant').reset_index()
        
        # Group by participant
        grouped_df = selected_df.groupby('participant')
        
        # Plot the grouped data on the current subplot
        for participant, group_data in grouped_df:
            ax.plot(group_data.index, group_data[sensor], label=f'Participant: {participant}')
        
        ax.set_ylabel(sensor)
        ax.set_xlabel('Sample time')
        ax.set_title(f"Exercise: {exercise}")
        ax.legend()  # Add legend for participants
        
    plt.suptitle(sensor)

plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
