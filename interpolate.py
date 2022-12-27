import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import os
import csv

file_path = "/Users/jenniferchoi/Documents/ASR/data/test1datasensorfiles/CAmotion/CAsensor/"
df_list = os.listdir(file_path)
#gyro, mag, acc

#list for all acc, gyro, and mag values
a_list = []
#time start and finish variables
tmin = 0
tmax = 10000000000000

#median list
medians=[]

#medians found by rounding values from median list
acc_gyr_m=20
mag_m=100

#read file and compare to find max and min times
imu_data = []
for count, df in enumerate(df_list[0:-1]):
    # print(file.shape)
    # print(df.dtypes)
    imu_data.append(np.loadtxt(file_path+df,delimiter=","))
    # print(len(file[:,0]))
    if imu_data[count][0,0]>tmin:
        tmin = imu_data[count][0,0]
    if imu_data[count][-1,0]<tmax:
        tmax=imu_data[count][-1,0]
    a_list.append(imu_data)
    
print(tmin, tmax)

# find interval for median(one time only)
# for i in range(len(a_list)):
#     intervals=a_list[i][1:,0]-a_list[i][0:-1,0]
#     medians.append(np.median(intervals))

xnew_acc = np.arange(tmin, tmax,acc_gyr_m ) #type is array
xnew_mag = np.arange(tmin, tmax, mag_m)

# plt.plot(xnew_gyr)
    
#interpolate

#make sure to add data at beginning, not inside for loop. or else replaced w zeros each time
data=np.zeros((xnew_acc.size, 10))

#need to do this after finding tmin and tmax
for count, df in enumerate(df_list[0:-1]):
    x = imu_data[count][:,0]
    
    #create new array filled with zeros for each axis data
    data[:,0]=xnew_acc
        
    #interpolate each of the axis values(x, y, z)
    for i in range(1,4):
        y = imu_data[count][:,i]
        f = interpolate.interp1d(x, y, kind='slinear')
        if df.startswith('acc') or df.startswith('gyr'):
            ynew = f(xnew_acc)
            xnew = xnew_acc
            data[:,3*count+i]= ynew
            # print(ynew[:5], data[:5,3*count+i])

        else:
            ynew = f(xnew_mag) 
            xnew = xnew_mag
            data[::5,3*count+i]= ynew
            
    # plot(do when you want to visualize data. not always necessary)
        plt.figure(figsize=(35,20))
        plt.plot(x, y, '-o', xnew, ynew, '-*')
        plt.title(df,fontsize=30)
        plt.xticks(np.arange(tmin, tmax, step=196))  # Set label locations.
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("Epoch(ms)",fontsize=20)
        if df.startswith("acc"):
            plt.ylabel("Accelerometer value(g)",fontsize=20)
        elif df.startswith("gyr"):
            plt.ylabel("Gyroscope value(degrees/second)", fontsize=20)
        else:
            plt.ylabel("Magnetometer value(microtesla)", fontsize=20)
        plt.show()

final_df = pd.DataFrame(data)
#change this later
final_df.to_csv("final_interpolated_cubic.csv")  
