import numpy as np
import pandas as pd
import torch
from scipy.signal import resample

from .chaotic_pde import blockify, generate_lorenz


def shuttle(file_path, seperator=" ", skipped_classes=[4], nominal_class=1):
    """Load the NASA space shuttle valve data set dataset.

    See B. Ferrell and S. Santuro. (2005). NASA Shuttle Valve Data. [Online].
    Available: http://www.cs.fit.edu/~pkc/nasa/data/
    Preprocess according to M. Munir, S. A. Siddiqui, A. Dengel and S. Ahmed,
    "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series,"
    in IEEE Access, vol. 7, pp. 1991-2005, 2019, doi: 10.1109/ACCESS.2018.2886457.
    1. Remove the samples with class value skipped_classes = [4]
    2. Mark datapoints of class nominal_class = 1 as nominal, the rest as anomal

    Returns:
        dataset [pd.dataframe]: Preprocessed shuttle dataset values
    """
    df = pd.read_csv(file_path, sep=seperator, engine="python", header=None)
    for j in range(df.shape[1]):
        df = df.rename(columns={j: "value_" + str(j)})
    df = df.rename(columns={"value_" + str(j): "is_anomaly"})
    df = df.dropna()
    df = df.astype(np.float32)
    for skipped_class in skipped_classes:
        df = df[df["is_anomaly"] != skipped_class]

    df["is_anomaly"] = df["is_anomaly"] != nominal_class
    return df


def discord_pkl(file_path):
    """Load from the discord dataset that has been downloaded an labeled by the downloader.

    For example on /data/discord/space_shuttle/labeled/train/TEK14.pkl

    Returns:
        pd.DataFrame: a pandas dataframe with colums value (float), is_anomaly (bool)
    """
    p = pd.read_pickle(file_path)
    df = pd.DataFrame(p)
    for j in range(df.shape[1]):
        df = df.rename(columns={j: "value_" + str(j)})
    df = df.rename(columns={"value_" + str(j): "is_anomaly"})
    df = df.astype(np.float32)
    df = df.astype({"is_anomaly": bool})
    return df


def yahoo_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"anomaly": "is_anomaly"})
    df = df.astype({"is_anomaly": bool})
    return df


def lorenz_data(
    length=1000,
    delta_t=0.01,
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
    rnd=True,
    normalize=True,
    anomalies=False,
    block_length=20,
    block_prop=0.2,
):
    spikes, _ = generate_lorenz(length, delta_t, sigma, beta, rho, 1, rnd, normalize)
    spikes = torch.squeeze(spikes)
    if anomalies:
        spikes, is_anomaly = blockify(
            spikes, block_length=block_length, block_prob=block_prop
        )

    df = pd.DataFrame(spikes).astype("float")
    df = df.rename(columns={0: "value_0"})
    if not anomalies:
        df["is_anomaly"] = False
    else:
        df["is_anomaly"] = is_anomaly
    return df

def nab(file_path, label_dir="./data/nab/labels/", seperator=","):
    df = pd.read_csv(file_path, sep=seperator, engine="python")
    df = df.rename(columns={"value" : "value_0"})
    with open(label_dir + "combined_windows.json") as label_file:
        split = file_path.split("/")
        name = split[-2] + "/" + split[-1]
        df_labels = json.load(label_file)
        bool_labels = np.zeros(df["value_0"].shape, dtype=bool)
        ts = df["timestamp"].values
        for window in df_labels[name]:
            bool_labels[np.logical_and(window[0] <= ts, window[1] >= ts)] = True
        df["is_anomaly"] = bool_labels
    return df

def get_values(df):
    return df[[c for c in df.columns if c.startswith("value")]]