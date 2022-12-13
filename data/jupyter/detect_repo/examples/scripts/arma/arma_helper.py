import arma_scipy
import data_loader

from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# res, scores, mystery_array = arma_scipy.fit(concat_data[:], [1,1], verbose=False)
# res_x.append(res.x)


def get_filepath(name, coeff_qp):
    filepath = Path(__file__)
    filepath = filepath.parent
    folder = str(filepath) + "/arma_estimations/"
    filepath = folder + name + "_" + str(coeff_qp) + ".pkl"
    return filepath


def estimate(coeff_qp=[1, 1], save=True, load=False):
    data = data_loader.load_sensor_data()
    res_x = {}
    for (name, dataset) in data.items():
        if load:
            filepath = get_filepath(name, coeff_qp)
            res_x[name] = pickle.load(open(filepath, "rb"))
        else:
            data = np.array(dataset[:])
            for i in range(data.shape[0]):
                if np.linalg.norm(data[i, :]) == 0:
                    continue
                # data[i, :] = data[i, :] / np.linalg.norm(data[i, :])
            res, scores = arma_scipy.fit(
                data, order=coeff_qp, verbose=False, solver="Powell", max_iter=100000
            )
            res_x[name] = res.x
            if save:
                filepath = get_filepath(name, coeff_qp)
                pickle.dump(res_x[name], open(filepath, "wb"))
    return res_x


def visualize(estimates, title):
    model = PCA(n_components=3, whiten=True)
    all_data = list(estimates.values())
    all_data = np.array(all_data[:])
    transformed = model.fit_transform(all_data)
    axes = plt.axes(projection="3d")
    for (name, estimate) in estimates.items():
        estimate = np.array(estimate)
        estimate = estimate.reshape(1, -1)
        data = model.transform(estimate)
        if name.find("nominal") > -1:
            axes.scatter3D(data[0, 0], data[0, 1], data[0, 2], marker="o")
        else:
            axes.scatter3D(data[0, 0], data[0, 1], data[0, 2], marker="s")
    axes.set_title(title)
    plt.show()


def print_avg_distance(estimates, title):
    ax = plt.axes()
    all_data = list(estimates.values())
    n = len(all_data) - 1
    i = 0
    for (name, estimate) in estimates.items():
        i += 1
        avg_dis = 0
        for other_estimate in estimates.values():
            avg_dis += np.linalg.norm(estimate - other_estimate)
        avg_dis /= n
        if name.find("nominal") > -1:
            ax.bar(str(i), avg_dis, color="xkcd:lightish blue")
        else:
            ax.bar(str(i), avg_dis, color="xkcd:light red")
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    coeff_qp = [2, 6]
    arma_estimates = estimate(coeff_qp=coeff_qp, save=True, load=False)
    visualize(arma_estimates, "ARMA Estimates with [q,p] = " + str(coeff_qp))
    print_avg_distance(
        arma_estimates, "Average Distances with [q,p] = " + str(coeff_qp)
    )
