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


def find_min_max_times(dfl: list[pd.DataFrame]) -> list:
    # time start and finish variables
    t_min: int = 0
    t_max: int = sys.maxsize

    # list for all acc, gyro, and mag values
    a_list = []
    imu_data = []
    for count, df in enumerate(dfl):
        imu_data = np.loadtxt(df, delimiter=",")
        print(imu_data.shape)
        print(imu_data[0, 0], imu_data[-1, 0])
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


def interpolate_signals(tmin: int, tmax: int, dfl: list[pd.DataFrame], a_list: list[np.ndarray]):
    xnew_acc = np.arange(tmin, tmax, ACC_INTERVAL)  # type is array
    xnew_mag = np.arange(tmin, tmax, MAG_INTERVAL)

    print("..........", xnew_acc.shape, xnew_mag.shape)

    # make sure to add data at beginning, not inside for loop. or else replaced w zeros each time
    data = np.zeros((xnew_acc.size, 1 + NB_SENSORS * NB_DOF * NB_AXES))
    print(a_list[0].shape)
    print(len(a_list))

    # create new array filled with zeros for each axis data
    data[:, 0] = xnew_acc

    # need to do this after finding tmin and tmax
    for count, df in enumerate(dfl):
        arr = remove_duplicates(a_list[count])

        x = arr[:, 0]
        print(count, x[0], x[-1], arr.shape)
        if count == 2:
            for i in range(x.size - 1):
                if x[i + 1] == x[i]:
                    import pdb;
                    pdb.set_trace()

        # interpolate each of the axis values(x, y, z)
        for i in range(1, 4):
            y = arr[:, i]
            f = interpolate.interp1d(x, y, kind='slinear')
            if df.startswith('acc') or df.startswith('gyr'):
                ynew = f(xnew_acc)
                xnew = xnew_acc
                data[:, 3 * count + i] = ynew
                # print(ynew[:5], data[:5,3*count+i])

            else:
                ynew = f(xnew_mag)
                xnew = xnew_mag
                data[::5, 3 * count + i] = ynew

    return data

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
    a_list, t_min, t_max = find_min_max_times(df_list)

    print("....", t_min, t_max)

    data = interpolate_signals(t_min, t_max, df_list, a_list)

    final_df = pd.DataFrame(data)
    # change this later
    final_df.to_csv("final_interpolated_test.csv")
