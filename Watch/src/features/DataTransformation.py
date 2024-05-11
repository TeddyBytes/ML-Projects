##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 22-12-2022

from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd


"""
LowPassFilter class for filtering data in a pandas DataFrame.

This class provides a method to apply a low-pass filter to a specific column in a pandas DataFrame.

Attributes:
    None

Methods:
    low_pass_filter(self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True)
        Applies a low-pass filter to a column in a pandas DataFrame.

        Args:
            data_table (pandas.DataFrame): The DataFrame containing the data to be filtered.
            col (str): The name of the column containing the signal data to be filtered.
            sampling_frequency (float): The sampling frequency of the data (in Hz).
            cutoff_frequency (float): The cut-off frequency of the low-pass filter (in Hz). This frequency represents the highest frequency component that will be preserved after filtering.
            order (int, optional): The order of the low-pass filter. Defaults to 5. Higher order filters provide sharper transitions but can also introduce more processing time.
            phase_shift (bool, optional): Controls whether to use `filtfilt` or `lfilter` for filtering. Defaults to True.
                * `filtfilt`: Applies the filter twice in opposite directions to minimize phase shift. This is recommended for most cases.
                * `lfilter`: Applies the filter in one direction, potentially introducing a phase shift. Use this option if preserving the original signal phase is crucial.

        Returns:
            pandas.DataFrame: The modified DataFrame with a new column named "{col}_lowpass" containing the filtered data.

Raises:
    None
"""


# This class removes the high frequency data (that might be considered noise) from the data.
# We can only apply this when we do not have missing values (i.e. NaN).
class LowPassFilter:
    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):

        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col] = filtfilt(b, a, data_table[col])
        else:
            data_table[col] = lfilter(b, a, data_table[col])
        # if phase_shift:
        #     data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        # else:
        #     data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table


# Class for Principal Component Analysis. We can only apply this when we do not have missing values (i.e. NaN).
# For this we have to impute these first, be aware of this.
class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table
