from pathlib import Path
from os import walk

import numpy as np
import datetime
import sys

import csv


class ft_loader:
    def extract_identifier(self, filename):
        split_filename = filename.split("_")
        identifier = split_filename[1]
        identifier += "_"
        identifier += split_filename[2].split(".")[0]
        return identifier

    def extract_seconds(self, ts):
        ts = ts.split("T")[1]
        ts = ts.split(".")[0]
        ts = ts.split(":")
        ts = 3600 * int(ts[0]) + 60 * int(ts[1]) + int(ts[2])
        return ts

    def read_timeseries_sensor(self, filename):
        with open(
            filename,
            newline="",
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            keys = reader.__next__()
            values = reader.__next__()
            arrays = []
            for i in range(8):
                arrays.append(np.array([values[i]], int))
            this_ts = self.extract_seconds(values[8])
            arrays.append(np.array(this_ts, int))
            for row in reader:
                next_ts = self.extract_seconds(row[8])
                if next_ts <= this_ts:
                    continue
                while next_ts - this_ts > 1:
                    this_ts += 1
                    for i in range(8):
                        arrays[i] = np.append(arrays[i], int(arrays[i][-1]))
                    arrays[8] = np.append(arrays[8], this_ts)
                for i in range(8):
                    arrays[i] = np.append(arrays[i], int(row[i]))
                arrays[8] = np.append(arrays[8], next_ts)
                this_ts = next_ts
        return arrays

    def read_timeseries_state(self, filename):
        with open(
            filename,
            newline="",
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            keys = reader.__next__()
            values = reader.__next__()
            arrays = []
            this_ts = self.extract_seconds(values[2])
            arrays.append(np.array([values[0]]))
            arrays.append(np.array(this_ts - 1))
            arrays[0] = np.append(arrays[0], values[1])
            arrays[1] = np.append(arrays[1], this_ts)
            for row in reader:
                next_ts = self.extract_seconds(row[2])
                if next_ts <= this_ts:
                    continue
                while next_ts - this_ts > 1:
                    this_ts += 1
                    arrays[0] = np.append(arrays[0], arrays[0][-1])
                    arrays[1] = np.append(arrays[1], this_ts)
                arrays[0] = np.append(arrays[0], row[1])
                arrays[1] = np.append(arrays[1], next_ts)
                this_ts = next_ts
        return arrays

    def pad_series(self, series, min_t, max_t):
        series[-1] = series[-1] - min_t
        for i in range(series[-1][0] - 1, -1, -1):
            for j in range(len(series) - 1):
                series[j] = np.append(series[j][0], series[j])
            series[-1] = np.append(i, series[-1])
        for i in range(series[-1][-1] + 1, max_t - min_t + 1):
            for j in range(len(series) - 1):
                series[j] = np.append(series[j], series[j][-1])
            series[-1] = np.append(series[-1], i)
        return series


import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse


def read_one_dataset(data_folder, plot=False):
    for (dirpath, dirnames, filenames) in walk(data_folder):
        loader = ft_loader()
        series_name = dirpath.split("/")[-1]
        loop = (x for x in filenames if x.endswith("csv"))
        data_dict = {}
        min_t = sys.maxsize
        max_t = 0
        for filename in loop:
            identifier = loader.extract_identifier(filename)
            if filename.find("input") > -1:
                series = loader.read_timeseries_sensor(dirpath + "/" + filename)
                data_dict[identifier] = series
            else:
                series = loader.read_timeseries_state(dirpath + "/" + filename)
                data_dict[identifier] = series
            min_t = min(min_t, series[-1][0])
            max_t = max(max_t, series[-1][-1])

        all_states = np.ones([])
        for key, series in data_dict.items():
            if key.find("state") > -1:
                all_states = np.append(all_states, np.copy(series[0]))
            series = loader.pad_series(series, min_t, max_t)
        enc = OneHotEncoder()
        all_states = np.array(all_states)
        all_states = all_states.reshape(-1, 1)[1:]
        enc.fit(all_states)

        for key, series in data_dict.items():
            if key.find("input") > -1:
                if plot:
                    for i in range(8):
                        plt.plot(series[8], series[i])
                data_dict[key] = np.array(series)
            if key.find("state") > -1:
                s = np.array(series[0])
                s = s.reshape(-1, 1)
                v = enc.transform(s)
                v = v.toarray()
                v = v.transpose()
                v = np.array(v)
                data_dict[key] = v
                if plot:
                    for i in range(n.shape[1]):
                        plt.plot(series[1], v[:, i].toarray())
            if plot:
                plt.title(key)
                plt.show()
    return data_dict


def load_sensor_data():
    sensor_keys = ["input_vgr", "input_hbw", "input_mpo", "input_sld"]
    filepath = Path(__file__)
    filepath = filepath.parent
    data_folder = str(filepath) + "/training_factory_timeseries/"

    (dirpath, dirnames, filenames) = walk(data_folder).__next__()
    data = {}

    for dirname in dirnames:
        data_dict = read_one_dataset(data_folder + dirname)
        concat_data = []
        for key in sensor_keys:
            v = data_dict[key]
            concat_data.extend(v)
        concat_data = np.array(concat_data)
        data[dirname] = concat_data
    return data


if __name__ == "__main__":
    data = load_sensor_data()
