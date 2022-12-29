import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import csv
from csv import DictWriter

#list file names
file_path = "Documents/ASR/notebooks/interpolated data" 
file_list = os.listdir(file_path)
df_list = []

#add column names and add to list
for file in file_list[0:-1]:
    if (file==".DS_Store"):
        continue
    else:
        df = pd.read_csv(file_path+file, header=None)
        df.rename(columns={0: 'time', 1: 'x', 2: 'y', 3: 'z'}, inplace=True)
        df.to_csv(file+"_wcol.csv", index=False)
        df_list.append(file+"_wcol.csv")

#read each file w/ column names in df list
df1= pd.read_csv(df_list[0])
df2 = pd.read_csv(df_list[1])
df3 = pd.read_csv(df_list[2])

#merge
new_file = pd.merge_asof(df1, df2, on='time')
final_file = pd.merge_asof(new_file, df3, on='time')

#save as csv file
final_file.to_csv("test_merged.csv")
