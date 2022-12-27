from typing import List, Any

import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import os
import glob
import csv

# gyro, mag, acc
DATA_DIR = "./data/good/"
df_list = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# use fixed sample intervals (in ms) based on ODR of the sensors
ACC_INTERVAL = 20
GYR_INTERVAL = ACC_INTERVAL
MAG_INTERVAL = 100

NB_SENSORS = 3  # A, B, C
NB_DOF = 3  # acc, gyr, mag
NB_AXES = 3


def find_min_max_times(dfl: list[str]) -> tuple[list, int, int]:
    """ Find the minimum and maximum timestamps for all the sensor data
    to prepare for the interpolation.

    :param dfl: list of strings storing the locations of the sensor data files.
    :return: list of the dataframes storing the imu data and min and max timestamps
    """
    # time start and finish variables
    t_min: int = 0
    t_max: int = sys.maxsize

    # list for all acc, gyro, and mag values
    a_list: list[np.ndarray] = []
    for count, df in enumerate(dfl):
        imu_data = np.loadtxt(df, delimiter=",")

        if imu_data[0, 0] > t_min:
            t_min = imu_data[0, 0]
        if imu_data[-1, 0] < t_max:
            t_max = imu_data[-1, 0]

        a_list.append(imu_data)

    return a_list, t_min, t_max


def remove_duplicates(arr: np.ndarray):
    seen = set()
    dup = []
    for idx, e in enumerate(arr[:, 0]):
        if e not in seen:
            seen.add(e)
        else:
            dup.append(idx)

    return np.delete(arr, dup, axis=0)


def interpolate_signals(t_min: int, t_max: int, dfl: list[pd.DataFrame], raw_data: list[np.ndarray]):
    x_new_acc = np.arange(t_min, t_max, ACC_INTERVAL)  # type is array
    x_new_mag = np.arange(t_min, t_max, MAG_INTERVAL)

    # make sure to add data at beginning, not inside for loop. or else replaced w zeros each time
    interp_results = np.zeros((x_new_acc.size, 1 + NB_SENSORS * NB_DOF * NB_AXES))

    # timestamps at which the signals are interpolated
    interp_results[:, 0] = x_new_acc

    # need to do this after finding t_min and tmax
    for count, df in enumerate(dfl):
        arr = remove_duplicates(raw_data[count])
        x = arr[:, 0]
        csv_filename = os.path.split(df)[-1]

        # interpolate each of the axis values(x, y, z)
        for i in range(1, 4):
            y = arr[:, i]
            f = interpolate.interp1d(x, y, kind='slinear')
            if csv_filename.startswith('acc') or csv_filename.startswith('gyr'):
                x_new, y_new = x_new_acc, f(x_new_acc)
                interp_results[:, 3 * count + i] = y_new
            else:
                x_new, y_new = x_new_mag, f(x_new_mag)
                interp_results[::5, 3 * count + i] = y_new

    return interp_results

    # # plot(do when you want to visualize data. not always necessary)
    # plt.figure(figsize=(35, 20))
    # plt.plot(x, y, '-o', xnew, ynew, '-*')
    # plt.title(df, fontsize=30)
    # plt.xticks(np.arange(tmin, tmax, step=196))  # Set label locations.
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel("Epoch(ms)", fontsize=20)
    # if df.startswith("acc"):
    #     plt.ylabel("Accelerometer value(g)", fontsize=20)
    # elif df.startswith("gyr"):
    #     plt.ylabel("Gyroscope value(degrees/second)", fontsize=20)
    # else:
    #     plt.ylabel("Magnetometer value(microtesla)", fontsize=20)
    # plt.show()


if __name__ == "__main__":
    raw_imu_data, min_t, max_t = find_min_max_times(df_list)
    interpolated_data = interpolate_signals(min_t, max_t, df_list, raw_imu_data)

    final_df = pd.DataFrame(interpolated_data)
    # change this later
    os.makedirs('./data/preprocessed', exist_ok=True)
    final_df.to_csv("./data/preprocessed/final_interpolated_test.csv")
