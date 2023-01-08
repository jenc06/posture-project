from typing import List, Any

import pandas as pd
import numpy as np
import csv

interp_df = pd.read_csv("/Users/jenniferchoi/Git/asr-project-2023/data/preprocessed/final_interpolated_test.csv", header=None, index_col=False, skiprows=1)
interp_np = interp_df.to_numpy()


for idx, i in enumerate(range(0,interp_np.shape[0],25)):
    print("running")
    sliced_name = "./data/sliced/sliced_"+str(idx)+".csv"
    sliced = interp_np[i:i+25,:]

    sliced_df = pd.DataFrame(sliced)
    final = sliced_df.to_csv(sliced_name, index=False, header=False)
    
    
    
