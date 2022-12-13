import json
import os

import numpy as np
import pandas as pd

from .ds_loader import yahoo_data


class IterReader:
    """Helper class for different reader implementations.

    Allows iterating over the reader to get every entry per loop. Use:
    for dataframe, metadata in reader:
        ...
    dataframe a pandas dataframe with colums value (float), labels (bool), timestamp (timestamp)
    metadata a dict that contains at least ["name"]
    """

    def __init__(self, reader):
        self.reader = reader
        self.idx = 0

    def __next__(self):
        try:
            ds = self.reader.read_df(self.idx)
            metadata = self.reader.create_metadata(self.idx)
            self.idx += 1
            return ds, metadata
        except IndexError:
            raise StopIteration


class NabIter:
    """
    Reader for the Numenta Dataset.

    Get the dataset at <https://github.com/numenta/NAB/archive/master.zip>
    """

    def __init__(self, data_dir="./data/nab/data/", label_dir="./data/nab/labels/"):
        """Init Numenta dataset reader.

        Args:
            data_dir (str): the path to the data, default "./data/nab/data/"
            label_dir (str): the path to the labels, default "./data/nab/labels/"
        """
        self.paths = []
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".csv"):
                    self.paths.append(os.path.join(root, name))
        with open(label_dir + "combined_windows.json") as label_file:
            self.nab_labels = json.load(label_file)

    def read_df(self, idx):
        """Read the timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
              The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            A pandas.dataframe with the attributes
              "value" that contains the timeseries,
              "is_anomaly" that contains the labels and "timestamp".
        """
        ds = pd.read_csv(self.paths[idx])
        ds_labels = self.nab_labels[self.get_name(idx)]
        bool_labels = np.zeros(ds["value"].shape, dtype=bool)
        ts = ds["timestamp"].values
        for window in ds_labels:
            bool_labels[np.logical_and(window[0] <= ts, window[1] >= ts)] = True
        ds["is_anomaly"] = bool_labels
        return ds

    def get_name(self, idx):
        """Read the name of timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
              The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            The name of timeseries data with a specific index
        """
        split = self.paths[idx].split("/")
        return split[-2] + "/" + split[-1]

    def create_metadata(self, idx):
        """Create an object of metadata for the timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
            The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            dict: contains ["name"] = str of the timeseries
        """
        metadata = {}
        metadata["name"] = self.get_name(idx)
        metadata["group"] = metadata["name"].split("/")[0]
        metadata["filepath"] = self.paths[idx]

        return metadata

    def __iter__(self):
        """Made iteratable with the IterReader class."""
        return IterReader(self)


class YahooIter:
    """Reader for the Yahoo Dataset.

    Get the dataset at <https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70>
    """

    def __init__(
        self, data_dir="./data/yahoo/ydata-labeled-time-series-anomalies-v1_0/"
    ):
        """Init reader for the Yahoo dataset.

        Args:
            data_dir="./data/yahoo/ydata-labeled-time-series-anomalies-v1_0/" (str): the path to the data
        """
        self.paths = []
        for root, _, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".csv") and not name.endswith("_all.csv"):
                    self.paths.append(os.path.join(root, name))

    def read_df(self, idx):
        """Read the timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
              The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            A pandas.dataframe with the attributes
              "value" that contains the timeseries,
              "is_anomaly" that contains the labels and "timestamp".
        """
        return yahoo_data(self.paths[idx])

    def get_name(self, idx):
        """Read the name of timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
              The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            str: the name of timeseries data with a specific index
        """
        split = self.paths[idx].split("/")
        return split[-2] + "/" + split[-1]

    def create_metadata(self, idx):
        """Create an object of metadata for the timeseries data with a specific index.

        Args:
            idx (int): the index of the timeseries.
              The index is defined by the order in which the files are read in and has no other meaning
        Returns:
            dict: contains ["name"] = str of the timeseries
        """
        metadata = {}
        metadata["name"] = self.get_name(idx)
        metadata["group"] = metadata["name"].split("/")[0]
        return metadata

    def __iter__(self):
        """Made iteratable with the IterReader class.

        next returns:
            dataframe, metadata where
            dataframe a pandas dataframe with colums value (float), labels (bool), timestamp (timestamp)
            metadata a dict that contains at least ["name"]
        """
        return IterReader(self)
