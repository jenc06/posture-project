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
        except ValueError:
            print(y, y_pd, pred)

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


def run_mlp(epochs: int=15):
    if epochs == 0:
        return
    # Need to load the train and test data separately
    # sensor_transform = transforms.Compose([transforms.ToTensor()])
    posture_sensor_dataset_train = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv"))
    train_dataloader = DataLoader(posture_sensor_dataset_train, batch_size=16, shuffle=True, num_workers=0)

    posture_sensor_dataset_test = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv"))
    test_dataloader = DataLoader(posture_sensor_dataset_test, batch_size=4, shuffle=False, num_workers=0)

    # Make a MLP model
    my_model = MyMLP().to(device)

    learning_rate = 1e-3

    my_loss_fn = nn.CrossEntropyLoss()
    my_optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, my_model, my_loss_fn, my_optimizer)
        test_loop(test_dataloader, my_model, my_loss_fn)
    print("Done!")


def load_train_test_data():
    train_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv")
    train_df = pd.read_csv(train_data_csv)
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, -1].values

    test_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv")
    test_df = pd.read_csv(test_data_csv)
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


def run_xgboost_classifier(X_train, y_train, X_test, y_test):
    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # train a model and check using cross validation
    print("XGBoost using the raw data")
    xgb_model = XGBClassifier(objective="multi:softmax")
    xgb_model.fit(X_train, y_train)
    print("Cross validation scores:")
    print(cross_val_score(XGBClassifier(), X_train, y_train, cv=cv))

    # run prediction
    y_pred = xgb_model.predict(X_test)
    correct = 0
    correct += (y_pred == y_test).sum().item()
    correct /= y_test.shape[0]
    print(f"(XGBoost) Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n")


def run_randomforest_classifer(X_train, y_train, X_test, y_test):
    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # train a random forest model
    print("Random Forest using the raw data")
    rfc_model = RandomForestClassifier(max_depth=7, random_state=10)
    rfc_model.fit(X_train, y_train)
    print("Cross validation scores:")
    print(cross_val_score(RandomForestClassifier(max_depth=7, random_state=10), X_train, y_train, cv=cv))

    # run prediction
    y_pred = rfc_model.predict(X_test)
    correct = 0
    correct += (y_pred == y_test).sum().item()
    correct /= y_test.shape[0]
    print(f"(RandomForest) Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n")


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_train_test_data()
    run_mlp(epochs=10)
    run_xgboost_classifier(X_tr, y_tr, X_te, y_te)
    run_randomforest_classifer(X_tr, y_tr, X_te, y_te)