import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import re

from matplotlib import colors
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# contains all the interpolated files
PREPROCESSED_DATA_FOLDER = "./data/preprocessed/all/"


def select_acc(sensor_data):
    return np.hstack([sensor_data[:, :3], sensor_data[:, 9:12], sensor_data[:, 18:21]])
    # all x values, but take only first three

def select_gyr(sensor_data):
    return np.hstack([sensor_data[:, 3:6], sensor_data[:, 12:15], sensor_data[:, 21:24]])


def select_mag(sensor_data):
    return np.hstack([sensor_data[:, 6:9], sensor_data[:, 15:18], sensor_data[:, 24:27]])


# features are x, labels are y
def make_features(good_interp: np.ndarray, mild_interp: np.ndarray, bad_interp: np.ndarray, gyr_skip: bool = True):
    # use select functions to extract acc, mag, and gyro
    good_acc, mild_acc, bad_acc = select_acc(good_interp), select_acc(mild_interp), select_acc(bad_interp)
    good_gyr, mild_gyr, bad_gyr = select_gyr(good_interp), select_gyr(mild_interp), select_gyr(bad_interp)
    good_mag, mild_mag, bad_mag = select_mag(good_interp), select_mag(mild_interp), select_mag(bad_interp)

    good_ft = good_acc
    mild_ft = mild_acc
    bad_ft = bad_acc

    # x: features, y: labels
    x = np.vstack([good_ft, mild_ft, bad_ft])
    y = np.hstack(
        [0 * np.ones(good_ft.shape[0]),
         1 * np.ones(mild_ft.shape[0]),
         2 * np.ones(bad_ft.shape[0])]).T.astype(np.int64)

    return x, y


def combine_cls_data(cls: str, these_sub_ids) -> np.ndarray:
    data_all = []
    for sub_id in these_sub_ids:
        print("SUB ID:", sub_id)
        # print("PATH: ", os.path.join(PREPROCESSED_DATA_FOLDER, f"final_interpolated_{cls}_s_{sub_id:03}*.csv"))
        data_all += glob.glob(os.path.join(PREPROCESSED_DATA_FOLDER, f"final_interpolated_{cls}_s_{sub_id:03}"+"*.csv"))

    # print("DATA ALL", data_all)
    # combine all good data
    # make empty row with right size
    # have to stack data. if there is nothing on top, cannot stack. length has to be same to vertical length
    # loadtxt loads a text file into array
    # print("DATA ALL SHAPE", data_all[0])
    data_tmp = np.loadtxt(data_all[0], delimiter=',', skiprows=1, usecols=range(2, 28))
    # make empty numpy array where u are workign with first row and number of total columns from first file
    data_comb = np.empty((0, data_tmp.shape[1]))
    for data in data_all:
        # load into variable
        data_arr = np.loadtxt(data, delimiter=',', skiprows=1, usecols=range(2, 28))
        # vertically stack each file from data_all into data_comb
        data_comb = np.vstack([data_comb, data_arr])

    return data_comb


def run_pca(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca_ = pca.transform(x)
    y_ = y.astype(np.int64)

    return x_pca_, y_


def plot3d_embedding(X, y, elev=50, azim=50) -> None:
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()

    for name, label in [("Good", 0), ("Mild", 1), ("Bad", 2)]:
        txt_colors = {0: "purple", 1: "green", 2: "red"}
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean() + 10 * label,
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.9, edgecolor=txt_colors[label], facecolor=txt_colors[label]),
        )

    # Reorder the labels to have colors matching the cluster results
    # 0: purple (good), 1: green (mild), 2: red (bad)
    colors = cm.rainbow(np.linspace(0, 1, 3))
    y_tmp = colors[np.choose(y, [0, 1, 2]).astype(int)]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_tmp, edgecolor="w")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()


if __name__ == "__main__":

    sub_ids = [0,1,2,3, 4, 5,6,7,8,10,11,12]
    good_combined = combine_cls_data('good', sub_ids)
    mild_combined = combine_cls_data('mild', sub_ids)
    bad_combined = combine_cls_data('bad', sub_ids)
    X_final, y_final = make_features(good_combined, mild_combined, bad_combined)

    print(X_final.shape, y_final.shape)

    print("Running PCA")
    X_pca, y = run_pca(X_final, y_final)
    # print("Running MDS")
    # X_mds, y = run_mds(X_final, y_final)
    # print("Running t-SNE")
    # X_tsne, y = run_tsne(X_final, y_final)

    plot3d_embedding(X_pca, y)
    # plot3d_embedding(X_mds, y)
    # plot3d_embedding(X_tsne, y)
