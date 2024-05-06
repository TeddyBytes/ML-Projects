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
# Plot single columns
# --------------------------------------------------------------


plt.plot(df[df["set"] == 1]["acc_y"].reset_index(drop=True))
plt.xlabel("Sample Time")
plt.plot(df[df["set"] == 2]["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------


exercises = df["exercise"].unique()
mpl.style.use("seaborn-v0_8-deep")


# Create the figure and subplots


for sensor in sensor_names.keys():
    # Loop through exercises and subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(40, 10))
    for i, exercise in enumerate(exercises):
        selected_df = df[df["exercise"] == exercise]

        # Access the current subplot using indexing
        ax = axs.flat[i]

        # Plot data on the current subplot
        ax.plot(selected_df[:100][sensor].reset_index(drop=True), label=exercise)

        ax.set_ylabel("Acc Y")
        ax.set_xlabel("Sample Time")
        ax.set_title(f"Exercise: {exercise_names[exercise]}")

    plt.suptitle(f"Exercise data as measured by {sensor_names[sensor]} sensor.")
    plt.savefig(f"../../reports/figures/testing.png")
    plt.show()

# --------------------------------------------------------------
# Define function to plot relevant data
# --------------------------------------------------------------


def make_ind_plt(df, sensor_data, groupby_sett):
    """Create subplots of each exercise for each sensor's data grouped by a specified setting.

    Args:
        df (DataFrame): The DataFrame containing the dataset.
        sensor_data (dict): A dictionary containing sensor names as keys and sensor details as values.
        groupby_sett (str): The column name based on which the data will be grouped.

    Returns:
        None: This function doesn't return anything. It generates plots directly.

    Example:
        makeplt(df, sensor_data, 'difficulty')

    This function generates subplots for each sensor's data grouped by the specified setting (e.g., 'difficulty').
    It iterates over each sensor, creates subplots for each exercise, and plots the grouped data.
    """

    # Define a color palette according to grouped plots
    palette = sns.color_palette("husl", len(df[groupby_sett].unique()))

    # Iterate over each sensor
    for sensor in sensor_data.keys():
        # Create a figure and subplots for the current sensor
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(40, 15))
        for ax, exercise in zip(axs.flat, exercise_names.keys()):

            # Filter data for the current exercise
            modified_df = (
                df[df["exercise"] == exercise]
                .sort_values(groupby_sett)
                .reset_index(drop=True)
            )
            grouped_df = modified_df.groupby(groupby_sett)
            for i, (group, grouped_data) in enumerate(grouped_df):
                # Plot the grouped data with a unique color from the palette
                ax.plot(
                    grouped_data[sensor],
                    label=f"{groupby_sett} : {group}",
                    color=palette[i],
                )
                ax.set_ylabel(sensor)
                ax.set_xlabel("Sample time")
                ax.set_title(f"Exercise: {exercise_names[exercise]}")
                ax.legend()
        # Save the figure
        plt.savefig(
            f"../../reports/figures/{sensor} exercise data by {groupby_sett}.png"
        )
        # Add suptitle for entire set of plots for this sensor
        plt.suptitle(f"Exercise data as measured by {sensor_names[sensor]} sensor.")


# --------------------------------------------------------------
# Compare medium vs. heavy sets for each exercise
# --------------------------------------------------------------

make_ind_plt(df, sensor_names, "difficulty")

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

make_ind_plt(df, sensor_names, "participant")

# --------------------------------------------------------------
# Compare sets
# --------------------------------------------------------------

make_ind_plt(df, sensor_names, "set")

# --------------------------------------------------------------
# Plot sensor data by exercise
# --------------------------------------------------------------


def plot_sensor_data_by_exercise(df):
    """Create subplots for split - sensor data grouped by exercise.

    Args:
        df (DataFrame): The DataFrame containing the dataset.
        sensor_data (dict): A dictionary containing sensor names as keys and sensor details as values.
        groupby_sett (str): The column name based on which the data will be grouped.
    Returns:
        None: This function doesn't return anything. It generates plots directly.

    Example:
        makeplt(df, sensor_data, 'difficulty')

    This function generates subplots for each sensor's data grouped by exercise.
    It iterates over each sensor, creates subplots for each exercise, and plots the grouped data.
    """

    grouped_sensors = [
        ["acc_x", "acc_y", "acc_z"],
        [
            "gyr_x",
            "gyr_y",
            "gyr_z",
        ],
    ]

    # Color list and alpha values for multiple lines (assuming same length as grouped_sensors)
    colors = plt.cm.get_cmap("tab10")(range(len(grouped_sensors[0])))
    alphas = [0.8, 0.7, 0.6]  # Adjust alpha values for desired transparency

    # Iterate over each exercise
    for exercise in exercise_names.keys():
        # Plot sensor data in triplets for each exercise
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(40, 15))
        # Filter data according to exercise
        modified_df = (
            df[df["exercise"] == exercise].sort_values("set").reset_index(drop=True)
        )
        # Iterate over each sensor
        for i, sensor in enumerate(grouped_sensors):
            # Plot each sensor with a different color and alpha
            for col, color, alpha in zip(sensor, colors, alphas):
                modified_df[col].plot.line(
                    ax=axs[i], label=col, color=color, alpha=alpha
                )
            axs[i].legend()
            axs[i].set_xlabel("Sample Time")
            if "acc_x" in sensor:
                axs[i].set_ylabel("Acceleration")
            else:
                axs[i].set_ylabel("Angular Velocity")
        # Save the figure
        plt.savefig(
            f"../../reports/figures/All sensor data for {exercise} exercise.png"
        )
        # Add suptitle for entire set of plots for this exercise
        plt.suptitle(f"Sensor data for the {exercise_names[exercise]} exercise.")

    plt.show()


# --------------------------------------------------------------
# Plot grouped sensor data by partipant
# --------------------------------------------------------------


def plot_sensor_data_by_participant(df):
    """Create subplots for split - sensor data grouped by participant.

    Args:
        df (DataFrame): The DataFrame containing the dataset.
    Returns:
        None: This function doesn't return anything. It generates plots directly.

    Example:
        makeplt(df)

    This function generates subplots for each sensor's data grouped by participant.
    It iterates over each sensor, creates subplots for each participant, and plots the grouped data.
    """

    grouped_sensors = [
        ["acc_x", "acc_y", "acc_z"],
        [
            "gyr_x",
            "gyr_y",
            "gyr_z",
        ],
    ]

    # Color list and alpha values for multiple lines (assuming same length as grouped_sensors)
    colors = plt.cm.get_cmap("tab10")(range(len(grouped_sensors[0])))
    alphas = [0.8, 0.7, 0.6]  # Adjust alpha values for desired transparency

    # Iterate over each exercise
    for exercise in exercise_names.keys():
        # Group data by participant within the exercise
        for participant in df[df["exercise"] == exercise]["participant"].unique():
            modified_df = df[
                (df["exercise"] == exercise) & (df["participant"] == participant)
            ].sort_values("set")
            modified_df = modified_df.reset_index(drop=True)

            # Plot sensor data in triplets for each participant within the exercise
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(40, 15))
            for i, sensor in enumerate(grouped_sensors):
                # Plot each sensor with a different color and alpha
                for col, color, alpha in zip(sensor, colors, alphas):
                    modified_df[col].plot.line(
                        ax=axs[i], label=col, color=color, alpha=alpha
                    )
                axs[i].legend()
                axs[i].set_xlabel("Sample Time")
                if "acc_x" in sensor:
                    axs[i].set_ylabel("Acceleration")
                else:
                    axs[i].set_ylabel("Angular Velocity")

            # Save the figure with exercise and participant information
            plt.savefig(
                f"../../reports/figures/All sensor data for {exercise_names[exercise]} exercise - Participant {participant}.png"
            )
            # Add suptitle for entire set of plots per exercise and participant
            plt.suptitle(
                f"Sensor data for the {exercise_names[exercise]} exercise - Participant {participant}"
            )

        plt.show()


# --------------------------------------------------------------
# Export all plots
# --------------------------------------------------------------

make_ind_plt(df, sensor_names, "difficulty")
make_ind_plt(df, sensor_names, "participant")
make_ind_plt(df, sensor_names, "set")

plot_sensor_data_by_exercise(df)
plot_sensor_data_by_participant(df)
