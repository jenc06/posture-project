import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset, DataLoader
from select_features import PREPROCESSED_DATA_FOLDER
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

def load_train_test_data(corrupt_test_data: bool = False):
    train_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv")
    train_df = pd.read_csv(train_data_csv)
    # frac=1 shuffle then return all rows of data
    train_df = train_df.sample(frac=1)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    test_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv")
    test_df = pd.read_csv(test_data_csv)
    test_df = test_df.sample(frac=1)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # made to check duplicate
    if corrupt_test_data:
        X_test[-10000:, :] = np.random.rand(10000, X_test.shape[1])

    return X_train, y_train, X_test, y_test

# make smt similar in 2d(image). my mLP into myCNN and try
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

    # returns length of csv file
    def __len__(self):
        return len(self.sensor_data_frame)

    # torch is type of data container.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            # convert index to list
            idx = idx.tolist()

        # use index idx to get columns from sensor data frame from index position 1 till end
        sensor_data = self.sensor_data_frame.iloc[idx, :-1]
        # make into numpy array
        sensor_data = np.array([sensor_data])
        # change data to 32-bit numbers
        sensor_data = sensor_data.astype('float32')

        # extract labels row by getting last column
        labels = self.sensor_data_frame.iloc[idx, -1]
        labels = np.array([labels])
        labels = labels.astype('int')

        sample = {'sensor_data': sensor_data, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RNNModel(nn.Module):
    def __init__(self, input_dim=18, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# batch_size, epoch and iteration
batch_size = 100
n_iters = 8000
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
posture_sensor_dataset_train = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv"))
train_dataloader = DataLoader(posture_sensor_dataset_train, batch_size=16, shuffle=True, num_workers=0)

posture_sensor_dataset_test = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv"))
test_dataloader = DataLoader(posture_sensor_dataset_test, batch_size=4, shuffle=False, num_workers=0)

# Create RNN
input_dim = 28  # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 1  # number of hidden layers
output_dim = 10  # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 28
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):

        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        if count % 250 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for sensor_data, labels in test_dataloader:
                images = Variable(images.view(-1, seq_dim, input_dim))

                # Forward propagation
                outputs = model(images)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += labels.size(0)

                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data_sample in enumerate(dataloader):
        # Compute prediction and loss

        X = data_sample['sensor_data']
        # torch.squeeze removes layer with size 1 dimension
        y = torch.squeeze(data_sample['labels'])

        # makes torch tensor filled with zeros
        y_p = torch.zeros_like(torch.empty(y.size(dim=0), 3))
        # set some values to 1
        y_p[torch.arange(y.size(dim=0)), y] = 1

        try:
            # move data X and y_pd to mps or cpu
            Xd = X.to(device)
            y_pd = y_p.to(device)
            # make predictions by running Xd through model
            pred = model(Xd)
            # find loss. pred is predictions. y_pd are true labels
            loss = loss_fn(pred, y_pd)
        except ValueError:
            print(y, y_pd, pred)

        # Backpropagation
        # set all gradients of parameters to zero
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update parameters based on new gradients
        optimizer.step()

        # print loss, current number of samples that have been processed
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # get total size
    size = len(dataloader.dataset)
    # get number of batches
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch no grad = no gradients calculated
    with torch.no_grad():
        for data_sample in dataloader:
            X = data_sample['sensor_data']
            y = torch.squeeze(data_sample['labels'])

            y_p = torch.zeros_like(torch.empty(y.size(dim=0), 3))
            y_p[torch.arange(y.size(dim=0)), y] = 1

            Xd = X.to(device)
            y_pd = y_p.to(device)
            pred = model(Xd)
            # loss_fn calculates loss between predicted and true values. add this to total test_loss
            test_loss += loss_fn(pred, y_pd).item()
            # compare tensors of predicted to true values
            # if 1.0 then prediction was correct add to correct variable
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    # get averages and print
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_train_test_data()
    # run_xgboost_classifier(X_tr, y_tr, X_te, y_te)
    # run_randomforest_classifer(X_tr, y_tr, X_te, y_te)
    run_mlp(epochs=10)