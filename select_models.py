import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from select_features import PREPROCESSED_DATA_FOLDER


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class PostureSensorDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sensor_data_frame = pd.read_csv(csv_file)
        # print(self.sensor_data_frame)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sensor_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print(idx)

        sensor_data = self.sensor_data_frame.iloc[idx, 1:]
        labels = self.sensor_data_frame.iloc[idx, -1]
        sensor_data = np.array([sensor_data])
        sensor_data = sensor_data.astype('float')
        sample = {'sensor_data': sensor_data, "labels": labels}
        print(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample


class MyMLP(nn.Module):
    def __init__(self, in_dim=18):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    # Need to load the train and test data separately
    posture_sensor_dataset = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv"))

    dataloader = DataLoader(posture_sensor_dataset, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['sensor_data'].size(), sample_batched['labels'].size())

    # Make a MLP model
    model = MyMLP().to(device)
    print(model)

    # cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #
    # print("XGBoost using the raw data")
    # print(cross_val_score(XGBClassifier(), X, y, cv=cv))
    # print("XGBoost using the PCA data")
    # print(cross_val_score(XGBClassifier(), X_pca, y, cv=cv))
    #
    # # print("SVM using the raw data")
    # # print(cross_val_score(svm.SVC(kernel='rbf', C=1, random_state=42), X, y, cv=cv))
    # # print("SVM using the PCA data")
    # # print(cross_val_score(svm.SVC(kernel='rbf', C=1, random_state=42), X_pca, y, cv=cv))
    #
    # print("RandomForeest using the raw data")
    # print(cross_val_score(RandomForestClassifier(max_depth=7, random_state=10), X, y, cv=cv))
    # print("RandomForeest using the PCA data")
    # print(cross_val_score(RandomForestClassifier(max_depth=7, random_state=10), X_pca, y, cv=cv))
    #
    # # print(cross_val_score(XGBClassifier(), X_mds, y))
    # # print("XGBoost using the t-SNE data")
    # # print(cross_val_score(XGBClassifier(), X_tsne, y))
    #
