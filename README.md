# Posture Project

  > Are you sitting like a shrimp? This can (theoretically) help you.

  A wearable ML pipeline that classifies posture — **good**, **mild**, or **bad** — using real IMU sensor data collected from human subjects. Built end-to-end: from streaming raw motion data over
  Bluetooth to training and comparing deep learning and classical models.

  ---

  ## How It Works

  Three **MetaWear** sensors are placed on the body and stream motion data wirelessly at 50Hz over BLE. Each sensor captures accelerometer, gyroscope, and magnetometer readings simultaneously. The data
  is merged, time-aligned, reduced with PCA, and fed into a suite of models to find what classifies posture best.

  Wear sensors → stream over BLE → align & clean → reduce with PCA → classify

  ---

  ## The Hardware

  **MetaWear by mbientlab** — 3 devices, worn on the body, connected over Bluetooth Low Energy.

  Each device runs three sensors in parallel:

  - **Accelerometer** (BMI270) — captures linear motion at ±4G
  - **Gyroscope** (BMI270) — captures rotation at ±1000 dps
  - **Magnetometer** (BMM150) — captures orientation relative to Earth's field

  All sensors sample at **50Hz**. Each session is tagged with a subject ID, trial ID, and posture label (`good` / `mild` / `bad`).

  ---

  ## The Pipeline

  **Collect**
  Stream all three sensors from all three devices simultaneously and save to CSV.

  ```bash
  python collect_imu_data.py --sub_id 001 --pose good --trial_id 000
  ```

  Merge & Clean
  Combine per-sensor files, extract the axes you want, then use interpolation to time-align readings across devices.
   ```bash
   python combine_columns.py
   python extract_dof.py
   python interpolate.py
  ```

  Reduce
  Apply PCA to compress the feature space. Visualize in 3D scatter plots or heat maps. Try alternatives (MDS, t-SNE, LLE) in the notebook.
  ```bash
  python select_features.py
  ```
  Train
  Benchmark 27 models at once with LazyPredict, then go deep on the best candidates.
   ```bash
  python lazypredict.py     # broad sweep
  python select_models.py   # MLP, XGBoost, Random Forest
  python cnn.py             # RNN
   ```
  ---
  The Models

  MLP — two hidden layers with LayerNorm, 50% dropout, ASGD optimizer, and early stopping so it knows when to quit. Trained with TensorBoard logging and per-class confusion matrices.

  XGBoost — multi-class softmax with 5-fold cross validation. Fast and competitive.

  Random Forest — 100 estimators, a solid baseline.

  RNN — single-layer recurrent model treating sensor readings as sequences.

  Leave-one-subject-out cross validation tests generalization across people, not just trials.

  ---
  Setup
  ```python
  pip install torch torchvision torchmetrics xgboost scikit-learn lazypredict pandas numpy matplotlib mbientlab-metawear
  ```
  ---
  Built With
  ```
  PyTorch · XGBoost · scikit-learn · LazyPredict · mbientlab MetaWear SDK
  ```
