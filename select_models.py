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
from torchvision import transforms
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
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sensor_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sensor_data = self.sensor_data_frame.iloc[idx, 1:]
        sensor_data = np.array([sensor_data])
        sensor_data = sensor_data.astype('float32')

        labels = self.sensor_data_frame.iloc[idx, -1]
        labels = np.array([labels])
        labels = labels.astype('int')

        sample = {'sensor_data': sensor_data, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MyMLP(nn.Module):
    def __init__(self, in_dim=18, out_dim=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data_sample in enumerate(dataloader):
        # Compute prediction and loss
        X = data_sample['sensor_data']
        y = torch.squeeze(data_sample['labels'])

        y_p = torch.zeros_like(torch.empty(y.size(dim=0), 3))
        y_p[torch.arange(y.size(dim=0)), y] = 1

        try:
            Xd = X.to(device)
            y_pd = y_p.to(device)
            pred = model(Xd)
            loss = loss_fn(pred, y_pd)
        except:
            print(y)
            print(y_pd)
            print(pred)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data_sample in dataloader:
            X = data_sample['sensor_data']
            y = torch.squeeze(data_sample['labels'])

            y_p = torch.zeros_like(torch.empty(y.size(dim=0), 3))
            y_p[torch.arange(y.size(dim=0)), y] = 1

            Xd = X.to(device)
            y_pd = y_p.to(device)
            pred = model(Xd)
            test_loss += loss_fn(pred, y_pd).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # Need to load the train and test data separately
    # sensor_transform = transforms.Compose([transforms.ToTensor()])
    posture_sensor_dataset_train = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv"))
    train_dataloader = DataLoader(posture_sensor_dataset_train, batch_size=16, shuffle=True, num_workers=0)

    posture_sensor_dataset_test = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv"))
    test_dataloader = DataLoader(posture_sensor_dataset_test, batch_size=4, shuffle=False, num_workers=0)

    # Make a MLP model
    model = MyMLP().to(device)

    learning_rate = 1e-3
    epochs = 50

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


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
