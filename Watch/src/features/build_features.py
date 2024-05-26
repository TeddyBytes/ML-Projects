import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(f"../../data/interim/02_outliers_removed_df.pkl")
df.info()
# sns.pairplot(df)

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
# Dealing with missing values (imputation)
# --------------------------------------------------------------

inter_df = df.copy()

# Interpolate null data via time method, preferred method for sequential / timeseries data.
inter_df = inter_df.interpolate("time")
inter_df.info()

# exer = inter_df.query(f"set == 20")
# exer[['acc_y','acc_z', 'acc_x' ]].reset_index(drop=True).plot()


# --------------------------------------------------------------
# Calculating average duration of set
# --------------------------------------------------------------
inter_df.sort_values(["time"], inplace=True)
inter_df["Duration"] = None

for set_num in inter_df["set"].unique():
    set_data = inter_df.query(f"set == {set_num}")
    time_data = set_data.index
    duration = time_data.max() - time_data.min()
    duration = duration.seconds

    inter_df.loc[inter_df["set"] == set_num, "Duration"] = duration


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# According to nyquist thm sampling frequency must be 2x max frequency


filtered_df = inter_df.copy()
filtered_df.info()
sample_freq = 5

lpf = LowPassFilter()

for sensor in sensor_names.keys():
    # Executes Inplace
    lpf.low_pass_filter(
        filtered_df,
        sensor,
        sample_freq,
        1.3,
    )

fig, axs = plt.subplots(
    nrows=6,
    ncols=2,
    figsize=(20, 15),
)

for i, sensor in enumerate(sensor_names.keys()):
    # Plot before/after lpf data
    axs[i][0].plot(
        inter_df[sensor][0:100].reset_index(drop=True),
    )
    axs[i][0].set_ylabel(sensor)
    axs[i][1].plot(
        filtered_df[sensor][0:100].reset_index(drop=True),
    )
    axs[i][1].set_ylabel(sensor)


# Labels and Title
axs[i][0].set_xlabel("Before LPF")
axs[i][1].set_xlabel("After LPF")
plt.savefig(f"../../reports/figures/Before and After LPF Data")

# Show plot
plt.show()


# --------------------------------------------------------------
# Test sampling frequency from FFT.
# --------------------------------------------------------------

fft_df = inter_df.copy()
fft_result = np.fft.fft(fft_df["acc_x"])

# Calculate the frequency bins
sampling_freq = (
    1 / (fft_df.index[1] - fft_df.index[0]).total_seconds()
)  # Sampling frequency in Hz
freq_bins = np.fft.fftfreq(len(fft_result), d=1 / sampling_freq)


# Plot frequency domain
plt.figure(figsize=(10, 6))
plt.plot(freq_bins, np.abs(fft_result))

# Labels and Title
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of acc_x")
plt.grid(True)

# Show Plot
plt.show()

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# Separate Numerical and Categorical features , PCA can only run on numerical data...
not_in_sensor_names = set(filtered_df.columns) - set(sensor_names.keys())
categorical_df = filtered_df[list(not_in_sensor_names)]
numerical_features = list(sensor_names.keys())
num_df = filtered_df.copy()[numerical_features]


# Try Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(num_df)

# Perform dimensionality reduction
pca = PCA(n_components=5)
pca_std_table = pca.fit_transform(standardized_data)

# Order of data is same as input data during standardization
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.sum(explained_variance)

# Plotting the scree plot
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o",
    linestyle="-",
)

# Labels and Title
plt.title("Scree Plot with Standardization Preprocessing.")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)

# Show plot
plt.show()


# Try Normalization

normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(num_df)

# Perform dimensionality reduction
# N_components selected via elbow method
pca = PCA(n_components=3)
pca_norm_table = pca.fit_transform(normalized_data)


# Order of data is same as input data during standardization
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.sum(explained_variance)

# Plotting the scree plot
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o",
    linestyle="-",
)

# Labels and Title
plt.title("Scree Plot with Normalization in preprocessing")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)

# Show plot
plt.show()


# Combine results into one df
pca_via_std_df = pd.DataFrame(
    pca_std_table,
    index=filtered_df.index,
    columns=[
        "pca_std_acc_x",
        "pca_std_acc_y",
        "pca_std_acc_z",
        "pca_std_gyr_x",
        "pca_std_gyr_y",
    ],
)

pca_via_norm_df = pd.DataFrame(
    pca_norm_table,
    index=filtered_df.index,
    columns=[
        "pca_norm_acc_x",
        "pca_norm_acc_y",
        "pca_norm_acc_z",
    ],
)

pca_df = pd.concat(
    [
        filtered_df,
        pca_via_std_df,
        pca_via_norm_df,
    ],
    axis=1,
)

pca_df.to_pickle(f"../../data/interim/03_PCA with norm and std.")

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

acc_r, gyr_r = 0, 0

for sensor in sensor_names.keys():
    if ("acc" in sensor) and ("pca" not in sensor):
        acc_r += pca_df[sensor] ** 2
    elif ("gyr" in sensor) and ("pca" not in sensor):
        gyr_r += pca_df[sensor] ** 2

pca_df["acc_r"] = np.square(acc_r)
pca_df["gyr_r"] = np.square(gyr_r)

pca_df.describe()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temp = pca_df.copy()
num_abs = NumericalAbstraction()
window_size = 5

features_to_abstract = list(sensor_names.keys()) + ["gyr_r", "acc_r"]
each_set_df = []

# Iterate over each exercise set
for ex_set in df_temp["set"].unique():
    # Filter by each set
    set_df = df_temp.query(f"set == {ex_set}").copy()
    # add rolling average/std dev/range to data
    abs_feature_df = num_abs.abstract_numerical(
        set_df, features_to_abstract, window_size, "mean"
    )
    abs_feature_df = num_abs.abstract_numerical(
        set_df, features_to_abstract, window_size, "std"
    )
    abs_feature_df = num_abs.calculate_range(set_df, features_to_abstract, window_size)
    each_set_df.append(abs_feature_df)

df_temp_ext = pd.concat(each_set_df)
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

fft_df = df_temp_ext.copy().reset_index()
fft = FourierTransformation()

each_set_df = []
for ex_set in fft_df["set"].unique():
    # Filter by each set
    # Reset index before copy, because df might be smaller after drop
    set_df = fft_df.query(f"set == {ex_set}").reset_index(drop=True).copy()
    set_df = fft.abstract_frequency(set_df, features_to_abstract, 14, 5)
    each_set_df.append(set_df)

df_fft = pd.concat(each_set_df).set_index("time", drop=True)

# Since we are dealing with highly correlated values, removing everyother datapoint reduces redundancy
df_fft = df_fft[::2]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

cluster_df = df_fft.copy()
subset = cluster_df[["acc_x", "acc_y", "acc_z"]].copy()
inertias = []
sillhouette = []

for k in range(2, 10):
    cluster = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster.fit(subset)
    inertias.append(cluster.inertia_)


x = np.arange(2, 2 + len(inertias))
plt.plot(x, inertias)

elbow = 4

final_cluster = KMeans(n_clusters=elbow, n_init=20, random_state=0)
final_cluster.fit(subset)

cluster_df["cluster"] = final_cluster.labels_

# Extracting centroids
centroids = final_cluster.cluster_centers_

# Plotting clusters in 3D with a larger figure size
fig = plt.figure(figsize=(14, 10))  # Adjust the figsize as needed
ax = fig.add_subplot(111, projection="3d")


# Plot each cluster with different colors and smaller dots
for exercise in cluster_df["exercise"].unique():
    ex_points = cluster_df[cluster_df["exercise"] == exercise]
    ax.scatter(
        ex_points["acc_x"],
        ex_points["acc_y"],
        ex_points["acc_z"],
        s=20,
        label=f"{exercise}",
    )

# Plot centroids with larger black dots
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    c="black",
    s=200,
    alpha=0.5,
    label="Centroids",
)

# Labels and title
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("3D K-Means Clustering")
ax.legend()

# Show plot
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

cluster_df.to_pickle("../../data/interim/04_FeatureEng_DF.pkl")
