import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA


def select_acc(sensor_data):
    return np.hstack([sensor_data[:, :3], sensor_data[:, 9:12], sensor_data[:, 18:21]])


def select_gyr(sensor_data):
    return np.hstack([sensor_data[:, 3:6], sensor_data[:, 12:15], sensor_data[:, 21:24]])


def select_mag(sensor_data):
    return np.hstack([sensor_data[:, 6:9], sensor_data[:, 15:18], sensor_data[:, 24:27]])


def make_features(good_interp: np.ndarray, mild_interp: np.ndarray, bad_interp: np.ndarray):
    good_acc, mild_acc, bad_acc = select_acc(good_interp), select_acc(mild_interp), select_acc(bad_interp)
    good_mag, mild_mag, bad_mag = select_mag(good_interp), select_mag(mild_interp), select_mag(bad_interp)
    good_gyr, mild_gyr, bad_gyr = select_gyr(good_interp), select_gyr(mild_interp), select_gyr(bad_interp)

    good_ft = np.hstack([good_acc, good_mag, good_gyr])
    mild_ft = np.hstack([mild_acc, mild_mag, mild_gyr])
    bad_ft = np.hstack([bad_acc, bad_mag, bad_gyr])

    # x: features, y: labels
    x = np.vstack([good_ft, mild_ft, bad_ft])
    y = np.hstack(
        [0 * np.ones(good_ft.shape[0]), 1 * np.ones(mild_ft.shape[0]), 2 * np.ones(bad_ft.shape[0])]).T.astype(np.int64)

    return x, y


def run_pca(x: np.ndarray, y: np.ndarray) -> tuple(np.ndarray, np.ndarray):
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca_ = pca.transform(x)
    y_ = y.astype(np.int64)

    return x_pca_, y_


def run_mds(x: np.ndarray, y: np.ndarray) -> tuple(nd.ndarray, np.ndarray):
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


def run_tsne(x: np.ndarray, y: np.ndarray) -> tuple(nd.ndarray, np.ndarray):
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


def plot3d_embedding(X, y, elev=50, azim=50) -> None:
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()

    for name, label in [("Good", 0), ("Mild", 1), ("Bad", 2)]:
        txt_colors = {0: "green", 1: "yellow", 2: "green"}
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean() + 10 * label,
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.9, edgecolor=txt_colors[label], facecolor=txt_colors[label]),
        )

    # Reorder the labels to have colors matching the cluster results
    y_tmp = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_tmp, edgecolor="w")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()


if __name__ == "__main__":
    # read the data (skip the first row and first two columns (1st: index, 2nd: timestamp))
    # skiprows deletes header(0,1,2...)
    # usecols range deletes time stamp column
    good_s0_t0 = np.loadtxt('./data/preprocessed/final_interpolated_good_s_002_t_001.csv', delimiter=',',
                            skiprows=1,
                            usecols=range(2, 29))
    mild_s0_t0 = np.loadtxt('./data/preprocessed/final_interpolated_mild_s_001_t_000.csv', delimiter=',', skiprows=1,
                            usecols=range(2, 29))
    bad_s0_t0 = np.loadtxt('./data/preprocessed/final_interpolated_bad_s_002_t_000.csv', delimiter=',', skiprows=1,
                           usecols=range(2, 29))

    x, y = make_features(good_s0_t0, mild_s0_t0, bad_s0_t0)

    x_pca, y = run_pca(x, y)
    plot3d_embedding(x_pca, y)

    x_mds, y = run_mds(x, y)
    plot3d_embedding(x_mds, y)

    x_tsne, y = run_mds(x, y)
    plot3d_embedding(x_tsne, y)
