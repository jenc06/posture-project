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
import warnings

# Change CLASS to the target class label to process the data associated with the given class
CLASSES = ['good', 'bad', 'mild']

# use fixed sample intervals (in ms) based on ODR of the sensors
# NOTE: Mag is sampled 5x slower than the accelerometer
ACC_INTERVAL = 20
GYR_INTERVAL = ACC_INTERVAL
MAG_INTERVAL = 100
STRIDE = int(MAG_INTERVAL/ACC_INTERVAL)

NB_SENSORS = 3  # A, B, C
NB_DOF = 3  # acc, gyr, mag
NB_AXES = 3 # x, y, z

# 'CA': A, 'DF': B, 'FF': C
SENSOR_ID = {'CA_C5_44_E0_3B_C3': 0, 'DF_D6_82_88_AF_42': 1, 'FF_EB_CA_C9_92_CF': 2}
DOF_ID = {'acc': 0, 'gyr': 1, 'mag': 2}


def find_min_max_times(dfl: list[str]) -> tuple[list, int, int]:
    """ Find the minimum and maximum timestamps for all the sensor data
    to prepare for the interpolation.

    :param dfl: list of strings storing the locations of the sensor data files.
    :return: list of the dataframes storing the imu data and min and max timestamps
    """
    if not dfl:
        raise Exception("No data file supplied.")

    # time start and finish variables
    t_min: int = 0
    t_max: int = sys.maxsize

    # list for all acc, gyro, and mag values
    a_list: list[np.ndarray] = []

    #find min and max times
    for count, df in enumerate(dfl):
        imu_data = np.loadtxt(df, delimiter=",")

        assert imu_data[0, 0] <= imu_data[-1, 0], "min time must be no greater than max time"

        if imu_data[0, 0] > t_min:
            t_min = imu_data[0, 0]
        if imu_data[-1, 0] < t_max:
            t_max = imu_data[-1, 0]

        a_list.append(imu_data)

    assert t_min <= t_max, "t_min must be no greater than t_max"

    return a_list, t_min, t_max


def extract_trial_ids(file_list: list[str]):
    trial_ids_ = set()
    for file in file_list:
        filename = os.path.split(file)[-1]
        start_ndx = filename.find('_t_')

        # NOTE: The trial id string size is assumed to be 3.
        # If the size changes, please change 6 to reflect the new size.
        trial_ids_.add(filename[start_ndx+3:start_ndx+6])
    return trial_ids_


def extract_subject_ids(file_list: list[str]):
    sub_ids_ = set()
    for file in file_list:
        filename = os.path.split(file)[-1]
        start_ndx = filename.find('_s_')

        # NOTE: The subject id string size is assumed to be 3.
        # If the size changes, please change 6 to reflect the new size.
        sub_ids_.add(filename[start_ndx + 3:start_ndx + 6])
    return sub_ids_


# take away repeating seconds. i.e.(1,2,2)->(1,2)
def remove_duplicates(arr: np.ndarray):
    seen = set()
    dup = []
    for idx, e in enumerate(arr[:, 0]):
        if e not in seen:
            seen.add(e)
        else:
            dup.append(idx)

    return np.delete(arr, dup, axis=0)


def get_sensor_id(filename: str) -> str:
    return filename[-21:-4]


def get_dof_id(filename: str) -> str:
    return filename[:3]


def interpolate_signals(t_min: int, t_max: int, dfl: list[pd.DataFrame], raw_data: list[np.ndarray]):
    x_new_acc = np.arange(t_min, t_max, ACC_INTERVAL)  # type is array
    x_new_mag = np.arange(t_min, t_max, MAG_INTERVAL)

    # make sure to add data at beginning, not inside for loop. or else replaced w zeros each time
    interp_results = np.zeros((x_new_acc.size, 1 + NB_SENSORS * NB_DOF * NB_AXES))

    # timestamps at which the signals are interpolated
    interp_results[:, 0] = x_new_acc

    # need to do this after finding t_min and tmax
    for count, df in enumerate(dfl):
        # function from before. arr would be one file from raw_imu_data(literally raw)
        arr = remove_duplicates(raw_data[count])
        # time stamps
        x = arr[:, 0]
        csv_filename = os.path.split(df)[-1]

        sensor_id = SENSOR_ID[get_sensor_id(csv_filename)]
        dof_id = DOF_ID[get_dof_id(csv_filename)]

        # calculate the index of the column to which the data is to be stored
        # The offset 1 is for the timestamp column.
        col_idx = sensor_id * (NB_DOF * NB_AXES) + dof_id * NB_DOF + 1

        # Recall that mag is sampled 5x slower than the accelerometer
        assert MAG_INTERVAL % ACC_INTERVAL == 0, \
            "Accelerometer interval should be divisible by magnetometer interval"

        # interpolate each of the axis values(x, y, z)
        for i in range(NB_AXES):
            y = arr[:, i+1]
            f = interpolate.interp1d(x, y, kind='slinear')
            if csv_filename.startswith('acc') or csv_filename.startswith('gyr'):
                x_new, y_new = x_new_acc, f(x_new_acc)
                interp_results[:, col_idx+i] = y_new
            else:
                x_new, y_new = x_new_mag, f(x_new_mag)
                interp_results[::STRIDE, col_idx+i] = y_new

    return interp_results


def plot_interpolated_data(min_ts: int, max_ts: int,
                           x: np.ndarray, y: np.ndarray,
                           x_new: np.ndarray, y_new: np.ndarray,
                           data_filename: str) -> None:
    # plot(do when you want to visualize data. not always necessary)
    data_filename = os.path.split(data_filename)[-1]
    plt.figure(figsize=(15, 10))
    plt.plot(x, y, '-o', x_new, y_new, '-*')
    plt.title(data_filename, fontsize=10)
    plt.xticks(np.arange(min_ts, max_ts, step=196))  # Set label locations.
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel("Epoch(ms)", fontsize=5)
    if data_filename.startswith("acc"):
        plt.ylabel("Accelerometer value(g)", fontsize=10)
    elif data_filename.startswith("gyr"):
        plt.ylabel("Gyroscope value(degrees/second)", fontsize=10)
    else:
        plt.ylabel("Magnetometer value(microtesla)", fontsize=10)
    plt.show()


def test_visualization(df_list_test: list[str], interp_data) -> None:
    # visualize the results
    # change index for different degree of freedom
    test_file = df_list_test[0]
    raw_data_test = np.loadtxt(test_file, delimiter=",")
    csv_filename_test = os.path.split(test_file)[-1]

    # calculate the index of the column in which the data is stored
    print(f"Loading csv file: {csv_filename_test}")
    sensor_idx = SENSOR_ID[get_sensor_id(csv_filename_test)]  # sensor A
    dof_idx = DOF_ID[get_dof_id(csv_filename_test)]  # accelerometer
    vis_axis = 'x'
    axis_idx = {'x': 0, 'y': 1, 'z': 2}
    col_ndx = sensor_idx * (NB_DOF * NB_AXES) + dof_idx * NB_AXES + 1

    x_vis, y_vis = raw_data_test[:, 0], raw_data_test[:, axis_idx[vis_axis] + 1]
    if get_dof_id(csv_filename_test) == 'mag':
        x_new_vis, y_new_vis = interp_data[::STRIDE, 0], interp_data[::STRIDE, col_ndx + axis_idx[vis_axis]]
    else:
        x_new_vis, y_new_vis = interp_data[:, 0], interp_data[:, col_ndx + axis_idx[vis_axis]]
    plot_interpolated_data(min_t, max_t, x_vis, y_vis, x_new_vis, y_new_vis, csv_filename_test)


if __name__ == "__main__":
    for cls in CLASSES:
        data_dir = f'./data/{cls}'
        df_list = glob.glob(os.path.join(data_dir, "*.csv"))
        print(df_list)

        trial_ids = extract_trial_ids(df_list)
        sub_ids = extract_subject_ids(df_list)

        for sub_id in sub_ids:
            for trial_id in trial_ids:
                df_list = glob.glob(os.path.join(data_dir, f"*_s_{sub_id}_t_{trial_id}*.csv"))
                if not df_list:
                    warnings.warn(f"No data available for sub id: {sub_id} and trial id: {trial_id}", RuntimeWarning)
                    continue

                raw_imu_data, min_t, max_t = find_min_max_times(df_list)
                interpolated_data = interpolate_signals(min_t, max_t, df_list, raw_imu_data)
                final_df = pd.DataFrame(interpolated_data)

                os.makedirs('./data/preprocessed', exist_ok=True)
                final_df.to_csv(f"./data/preprocessed/all/final_interpolated_{cls}_s_{sub_id}_t_{trial_id}.csv")

                # visualize the results
                # test_visualization(df_list, interpolated_data)
