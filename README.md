# asr-project-2023
cnn.py: CNN model to classify posture. 

collect_imu_data: takes sensor readings and saves them as a CSV file

combine_columns: combines columns of individual sensor readings into 1 CSV file

extract_dof.py: extract any combination of accelerometer, gyroscope, and/or magnetometer data from the CSV file

interpolate.py: use interpolation to time-align the data from different sensors.

lazypredict.py: use the lazypredict python package to run 27 different ML models and see their accuracies, runtime, balanced accuracy, and F1 score

PCA_cv_data_make: create CSV files for PCA cross validation where the training = 1 human subject and testing will be all other subjects+spine model

PCA_data_make: apply PCA to interpolated accelerometer data and save as training and testing CSV file. Visualize in 3d scatterplot.

prototype_features.ipynb: try different data transformation methods: PCA, MDS, TSNE, LLE(locally linear embedding). Visualize as a heat map

section_select_features.py: visualize a wanted section of data instead of the whole folder. Can pick certain subjects' data to graph on a 3d scatterplot

select_features.py: same as PCA_data_make but applied to accelerometer, gyroscope, and magnetometer data. 

select_models.py: tests various machine learning models. Includes a dataloader to extract x and y, an MLP model, CNN model, Resnet model, xgboost classifier model, and random forest classifier model. MLP model includes an early stopping function.

slice_times.py: split into sections of 25 rows of data and saved as individual CSV files. Did not use in final project


