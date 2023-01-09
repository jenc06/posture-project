import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


PREPROCESSED_DATA_FOLDER = "./data/preprocessed/"


def select_acc(sensor_data):
    return np.hstack([sensor_data[:, :3], sensor_data[:, 9:12], sensor_data[:, 18:21]])


def select_gyr(sensor_data):
    return np.hstack([sensor_data[:, 3:6], sensor_data[:, 12:15], sensor_data[:, 21:24]])


def select_mag(sensor_data):
    return np.hstack([sensor_data[:, 6:9], sensor_data[:, 15:18], sensor_data[:, 24:27]])


def make_features(good_interp: np.ndarray, mild_interp: np.ndarray, bad_interp: np.ndarray, gyr_skip: bool=True):
    good_acc, mild_acc, bad_acc = select_acc(good_interp), select_acc(mild_interp), select_acc(bad_interp)
    good_mag, mild_mag, bad_mag = select_mag(good_interp), select_mag(mild_interp), select_mag(bad_interp)
    good_gyr, mild_gyr, bad_gyr = select_gyr(good_interp), select_gyr(mild_interp), select_gyr(bad_interp)

    good_ft = np.hstack([good_acc, good_mag, good_gyr])
    mild_ft = np.hstack([mild_acc, mild_mag, mild_gyr])
    bad_ft = np.hstack([bad_acc, bad_mag, bad_gyr])

    if gyr_skip:
        good_ft = np.hstack([good_acc, good_mag])
        mild_ft = np.hstack([mild_acc, mild_mag])
        bad_ft = np.hstack([bad_acc, bad_mag])

    # x: features, y: labels
    x = np.vstack([good_ft, mild_ft, bad_ft])
    y = np.hstack(
        [0 * np.ones(good_ft.shape[0]), 1 * np.ones(mild_ft.shape[0]), 2 * np.ones(bad_ft.shape[0])]).T.astype(np.int64)

    return x, y


def run_pca(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca_ = pca.transform(x)
    y_ = y.astype(np.int64)

    return x_pca_, y_


def run_mds(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # MDS
    md_scaling = manifold.MDS(
        n_components=3,
        max_iter=50,
        n_init=4,
        random_state=0,
        normalized_stress=False,
    )
    x_mds_ = md_scaling.fit_transform(x)
    y_ = y.astype(np.int64)

    return x_mds_, y_


def run_tsne(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_sne = manifold.TSNE(
        n_components=3,
        perplexity=30,
        init="random",
        n_iter=250,
        random_state=0,
    )
    x_tsne_ = t_sne.fit_transform(x)
    y_ = y.astype(np.int64)

    return x_tsne_, y_


def combine_cls_data(cls: str) -> np.ndarray:
    # combine all good data
    data_all = glob.glob(os.path.join(PREPROCESSED_DATA_FOLDER, f"final_interpolated_{cls}_*.csv"))
    data_tmp = np.loadtxt(data_all[0], delimiter=',', skiprows=1, usecols=range(2, 29))

    data_comb = np.empty((0, data_tmp.shape[1]))
    for data in data_all:
        data_arr = np.loadtxt(data, delimiter=',', skiprows=1, usecols=range(2, 29))
        data_comb = np.vstack([data_comb, data_arr])

    return data_comb


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
    # read the data (skip the first row and first two columns (1st: index, 2nd: timestamp))
    # skiprows deletes header(0,1,2...)
    # usecols range deletes time stamp column

    # combine each class data
    good_combined = combine_cls_data('good')
    mild_combined = combine_cls_data('mild')
    bad_combined = combine_cls_data('bad')

    # good_comb_train, good_comb_val, good_comb_test = combine_cls_data('good')

    X, y = make_features(good_combined, mild_combined, bad_combined)

    print("Running PCA")
    X_pca, y = run_pca(X, y)
    # print("Running MDS")
    # X_mds, y = run_mds(X, y)
    print("Running t-SNE")
    # X_tsne, y = run_tsne(X, y)

    # plot3d_embedding(X_pca, y)

    print("XGBoost using the raw data")
    print(cross_val_score(XGBClassifier(), X, y))
    print("XGBoost using the PCA data")
    print(cross_val_score(XGBClassifier(), X_pca, y))
    # print(cross_val_score(XGBClassifier(), X_mds, y))
    print("XGBoost using the t-SNE data")
    print(cross_val_score(XGBClassifier(), X_tsne, y))

    # x_mds, y = run_mds(x, y)
    # plot3d_embedding(x_mds, y)
    #
    # x_tsne, y = run_mds(x, y)
    # plot3d_embedding(x_tsne, y)
