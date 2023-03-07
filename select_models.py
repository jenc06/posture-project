import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn.functional as F
from select_features import PREPROCESSED_DATA_FOLDER
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.optim.lr_scheduler import MultiplicativeLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter


device = "cpu" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_test_loss = np.inf

    def early_stop(self, test_loss):
        if test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
            self.counter = 0
        elif test_loss > (self.min_test_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# make smt similar in 2d(image). my mLP into myCNN and try
class PostureSensorDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None, acc_only=False):
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
        self.acc_only = acc_only

    # returns length of csv file
    def __len__(self):
        return len(self.sensor_data_frame)

    # torch is type of data container.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            # convert index to list
            idx = idx.tolist()

        # use index idx to get columns from sensor data frame from index position 1 till end
        # make into numpy array
        if self.acc_only:
            sensor_data = self.sensor_data_frame.iloc[idx, :3]
        else:
            sensor_data = self.sensor_data_frame.iloc[idx, :-1]

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


class PostureSensorDataset2D(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None, multichannel=False, acc_only=False):
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
        self.num_ts = 10
        self.multichannel = multichannel
        self.acc_only = acc_only

    def __len__(self):
        return len(self.sensor_data_frame) // self.num_ts

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.acc_only:
            sensor_data = self.sensor_data_frame.iloc[idx * self.num_ts:(idx + 1) * self.num_ts, :3]
        else:
            sensor_data = self.sensor_data_frame.iloc[idx * self.num_ts:(idx + 1) * self.num_ts, :-1]

        if self.multichannel:
            sensor_data = np.expand_dims(sensor_data, 2)
            sensor_data = np.tile(sensor_data, (1, 1, 3))
            sensor_data = np.transpose(sensor_data, [2, 0, 1])

        if not isinstance(sensor_data, np.ndarray):
            sensor_data = sensor_data.astype('float32').to_numpy()
        else:
            sensor_data = sensor_data.astype('float32')

        if self.acc_only:
            labels = self.sensor_data_frame.iloc[idx * self.num_ts:(idx + 1) * self.num_ts, -1]
        else:
            labels = self.sensor_data_frame.iloc[idx * self.num_ts:(idx + 1) * self.num_ts, -1]

        if all(labels == 0):
            # print("label 0")
            labels = 0
        elif all(labels == 1):
            labels = 1
        elif all(labels == 2):
            labels = 2
        else:
            # mixed data and labels
            labels = 3

        labels = np.array([labels])
        labels = labels.astype('int')

        sample = {'sensor_data': sensor_data, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MyMLP(nn.Module):
    def __init__(self, in_dim=9, out_dim=3, acc_only=True):
        # super is a function used to call the init class. all functions from init will run
        super().__init__()
        # make tensor 1D
        self.acc_only = acc_only
        self.flatten = nn.Flatten()

        if acc_only:
            in_dim = 3
        else:
            in_dim = 9

        # sequence of layers: linear layers apply linear transformation, ReLU is nonlinear activation function
        # LayerNorm = normalization layer. helps stabilize training
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits


# class MyCNN1D(nn.Module):
#     def __init__(self, in_dim=18, out_dim=3):
#         super().__init__()
#         # self.flatten = nn.Flatten()
#         self.conv1d_relu_stack = nn.Sequential(
#             nn.Conv1d(in_dim, 64, kernel_size=1),
#             # nn.BatchNorm1d(64),
#             # nn.ReLU(),
#             # nn.Conv1d(64, 128, kernel_size=3, padding="same"),
#             # nn.BatchNorm1d(128),
#             # nn.ReLU(),
#             # nn.Conv1d(128, 128, kernel_size=3, padding="same"),
#             # nn.BatchNorm1d(128),
#             # nn.ReLU(),
#             nn.Linear(64, out_dim),
#             # nn.ReLU(),
#         )
#
#     def forward(self, x):
#         # x = self.flatten(x)
#         x = torch.transpose(x, 1, 2)
#         logits = self.conv1d_relu_stack(x)
#         return logits
#
#
# # 3x3 convolution
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)
#
#
# # Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# # ResNet
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=4, acc_only=False):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(3, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.avg_pool = nn.AvgPool2d(5)
#         if acc_only:
#             self.fc = nn.Linear(32, num_classes)
#         else:
#             self.fc = nn.Linear(96, num_classes)
#
#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels))
#
#         layers = list()
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
#
#
# class MyCNN2D(nn.Module):
#     def __init__(self, in_dim=1, out_dim=4):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_dim, 6, 3)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.conv2 = nn.Conv2d(6, 32, 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, out_dim)
#         self.dropout = nn.Dropout(0.50)
#
#     def forward(self, x):
#         x = torch.transpose(x[None, :, :, :], 0, 1)
#         x = self.dropout(self.pool1(F.relu(self.bn1(self.conv1(x)))))
#         x = self.dropout(self.pool2(F.relu(self.bn2(self.conv2(x)))))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.fc3(x)
#
#         # print(x)
#         return x


def train_loop(dataloader, epoch, model, loss_fn, optimizer, scheduler=None, cnn2d=False, writer=None):
    size = len(dataloader.dataset)
    num_classes = 4 if cnn2d else 3
    sizes = [0] * num_classes
    corrects = [0] * num_classes
    best_loss = 1e10

    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    if num_classes == 3:
        cm = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    else:
        cm = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    for batch, data_sample in enumerate(dataloader):
        # Compute prediction and loss

        X = data_sample['sensor_data']
        # torch.squeeze removes layer with size 1 dimension
        y = torch.squeeze(data_sample['labels'])

        for n in range(num_classes):
            sizes[n] += (y == n).type(torch.float).sum().item()

        if cnn2d:
            y_p = torch.zeros_like(torch.empty(y.size(dim=0), 4))
        else:
            y_p = torch.zeros_like(torch.empty(y.size(dim=0), 3))

        y_p[torch.arange(y.size(dim=0)), y] = 1

        try:
            # move data X and y_pd to mps or cpu
            Xd = X.to(device)
            y_pd = y_p.to(device)
            # make predictions by running Xd through model
            pred = model(Xd)
            # find loss. pred is predictions. y_pd are true labels
            loss = loss_fn(pred, y_pd)
        except (ValueError, RuntimeError, TypeError) as e:
            # print(e, y, y_pd, pred)
            print(e)

        cm += metric(pred.argmax(1), y)

        for n in range(num_classes):
            corrects[n] += \
                torch.logical_and((pred.argmax(1) == y.to(device)), (y.to(device) == n)).type(torch.float).sum().item()

        # Backpropagation
        # set all gradients of parameters to zero
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update parameters based on new gradients
        optimizer.step()

        if not scheduler:
            scheduler.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if writer:
                writer.add_scalar('training loss', loss, epoch)
                writer.add_scalar('training acc', 100 * sum(corrects) / sum(sizes), epoch)

    print("training metric: \n", cm)


def test_loop(dataloader, epoch, model, loss_fn, cnn2d=False, writer=None):
    size = len(dataloader.dataset)
    # get number of batches
    num_batches = len(dataloader)
    num_classes = 4 if cnn2d else 3
    test_loss, correct = 0, 0
    sizes = [0] * num_classes
    corrects = [0] * num_classes

    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    if num_classes == 3:
        cm = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    else:
        cm = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # torch no grad = no gradients calculated
    with torch.no_grad():
        for data_sample in dataloader:
            X = data_sample['sensor_data']
            y = torch.squeeze(data_sample['labels'])

            try:
                y_p = torch.zeros_like(torch.empty(y.size(dim=0), num_classes))
                y_p[torch.arange(y.size(dim=0)), y] = 1
            except IndexError:
                continue

            for n in range(num_classes):
                sizes[n] += (y == n).type(torch.float).sum().item()

            Xd = X.to(device)
            y_pd = y_p.to(device)
            pred = model(Xd)
            # loss_fn calculates loss between predicted and true values. add this to total test_loss
            test_loss += loss_fn(pred, y_pd).item()
            cm += metric(pred.argmax(1), y)

            for n in range(num_classes):
                corrects[n] += \
                    torch.logical_and((pred.argmax(1) == y.to(device)), (y.to(device) == n)).type(torch.float).sum().item()

    # get averages and print
    test_loss /= num_batches
    correct = sum(corrects)
    size = sum(sizes)
    for n in range(num_classes):
        if sizes[n] > 0:
            corrects[n] /= sizes[n]

    for n in range(num_classes):
        print(f"Test Error: Accuracy: {(100 * corrects[n]):>0.1f}%, sz: {sizes[n]}, Avg loss: {test_loss:>8f}")
    print(f"Test Error: Accuracy: {(100 * correct/size):>0.1f}%, Avg loss: {test_loss:>8f}")

    if writer:
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('test acc', 100*correct/size, epoch)

    print("test metric: \n", cm)
    return test_loss


def run_mlp(epochs: int = 15, acc_only=True):
    if epochs == 0:
        return "epochs was 0"

    # Need to load the train and test data separately
    # sensor_transform = transforms.Compose([transforms.ToTensor()])
    posture_sensor_dataset_train = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data_pca.csv"),
                                                        acc_only=acc_only)
    train_dataloader = DataLoader(posture_sensor_dataset_train, batch_size=16, shuffle=True, num_workers=0)

    posture_sensor_dataset_test = PostureSensorDataset(os.path.join(PREPROCESSED_DATA_FOLDER, "test_data_pca.csv"),
                                                       acc_only=acc_only)
    test_dataloader = DataLoader(posture_sensor_dataset_test, batch_size=4, shuffle=False, num_workers=0)

    # Make a MLP model
    my_model = MyMLP().to(device)

    learning_rate = 1e-2

    # n_smp_cls = [14300, 11300, 9600]
    # wgt = torch.tensor(n_smp_cls) / sum(n_smp_cls)
    # my_loss_fn = nn.CrossEntropyLoss(weight=wgt)
    my_loss_fn = nn.CrossEntropyLoss()
    my_optimizer = torch.optim.ASGD(my_model.parameters(), lr=learning_rate)
    lmbda = lambda epoch: 0.9
    scheduler = MultiplicativeLR(my_optimizer, lr_lambda=lmbda)
    # scheduler = CosineAnnealingWarmRestarts(my_optimizer, 10)

    os.makedirs('./runs', exist_ok=True)
    if posture_sensor_dataset_train.acc_only:
        writer = SummaryWriter('runs/mlp_experiments_acc_only/')
    else:
        writer = SummaryWriter('runs/mlp_experiments_acc_mag/')

    best_loss = 1e9
    os.makedirs('./models', exist_ok=True)
    early_stopping=EarlyStopper(patience=20, min_delta=0.05)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, t, my_model, my_loss_fn, my_optimizer, scheduler=scheduler, writer=writer)
        test_loss = test_loop(test_dataloader, t, my_model, my_loss_fn, writer=writer)
        if early_stopping.early_stop(test_loss):
            print("we are at epoch:", t)
            break

        if test_loss < best_loss:
            if posture_sensor_dataset_train.acc_only:
                torch.save(my_model, './models/best-model-mlp-acc-pca.pt')
            else:
                torch.save(my_model, './models/best-model-mlp-acc-mag.pt')

    writer.close()

    print("Done!")


# def run_cnn2d(epochs: int = 15, arch='my_resnet', multichannel=False, acc_only=False):
#     if epochs == 0:
#         return
#     # Need to load the train and test data separately
#     # sensor_transform = transforms.Compose([transforms.ToTensor()])
#     posture_sensor_dataset_train = PostureSensorDataset2D(os.path.join(PREPROCESSED_DATA_FOLDER, "train_data_pca.csv"),
#                                                           multichannel=multichannel, acc_only=acc_only)
#     train_dataloader = DataLoader(posture_sensor_dataset_train, batch_size=16, shuffle=True, num_workers=0)
#
#     posture_sensor_dataset_test = PostureSensorDataset2D(os.path.join(PREPROCESSED_DATA_FOLDER, "test_data_pca.csv"),
#                                                          multichannel=multichannel, acc_only=acc_only)
#     test_dataloader = DataLoader(posture_sensor_dataset_test, batch_size=4, shuffle=False, num_workers=0)
#
#     # Make a 2D CNN
#     num_classes = 4
#     if arch == 'cnn2d':
#         my_model = MyCNN2D().to(device)
#     elif arch == 'resnet18_pretrained':
#         my_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         num_ftrs = my_model.fc.in_features
#         my_model.fc = nn.Linear(num_ftrs, num_classes)
#         my_model.to(device)
#     elif arch == 'my_resnet':
#         my_model = ResNet(ResidualBlock, [1], acc_only=posture_sensor_dataset_train.acc_only).to(device)
#     else:
#         raise Exception(f"Arch {arch} not supported.")
#
#     learning_rate = 1e-8 #6e-4
#     # learning_rate = 1e-2
#     best_loss = 1e9
#
#     n_smp_cls = [14300, 11300, 9600, 50]
#     wgt = torch.tensor(n_smp_cls) / sum(n_smp_cls)
#     my_loss_fn = nn.CrossEntropyLoss(weight=wgt)
#     my_optimizer = torch.optim.ASGD(my_model.parameters(), lr=learning_rate)
#     scheduler = CosineAnnealingWarmRestarts(my_optimizer, 20)
#     early_stopping = EarlyStopper(patience=5, min_delta=0.5)
#
#     os.makedirs('./runs', exist_ok=True)
#     if posture_sensor_dataset_train.acc_only:
#         writer = SummaryWriter(f'runs/{arch}_experiments_acc_only')
#     else:
#         writer = SummaryWriter(f'runs/{arch}_experiments_acc_mag')
#
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train_loop(train_dataloader, t, my_model, my_loss_fn, my_optimizer,
#                    scheduler=scheduler, writer=writer, cnn2d=True)
#         test_loss = test_loop(test_dataloader, t, my_model, my_loss_fn,
#                               writer=writer, cnn2d=True)
#
#         if early_stopping.early_stop(test_loss):
#             print("We are at epoch:", t)
#             break
#
#     os.makedirs('./models', exist_ok=True)
#     if test_loss < best_loss:
#         if posture_sensor_dataset_train.acc_only:
#             torch.save(my_model, f'./models/best-model-{arch}-acc-only.pt')
#         else:
#             torch.save(my_model, f'./models/best-model-{arch}-acc-mag.pt')
#
#     writer.close()
#     print("Done!")
#     # return my_loss_fn


def load_train_test_data(corrupt_test_data: bool = False, sub_id: int=0):
    # if sub_id == 0:
    #     train_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "train_data.csv")
    # else:
    #     train_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, f"train_data_{sub_id}.csv")
    train_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "train_data_pca.csv")
    train_df = pd.read_csv(train_data_csv)
    # frac=1 shuffle then return all rows of data
    train_df = train_df.sample(frac=1)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    # if sub_id == 0:
    #     test_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "test_data.csv")
    # else:
    #     test_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, f"test_data_{sub_id}.csv")
    test_data_csv = os.path.join(PREPROCESSED_DATA_FOLDER, "test_data_pca.csv")
    test_df = pd.read_csv(test_data_csv)
    test_df = test_df.sample(frac=1)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # made to check duplicate
    if corrupt_test_data:
        X_test[-10000:, :] = np.random.rand(10000, X_test.shape[1])

    return X_train, y_train, X_test, y_test


def run_xgboost_classifier(X_train, y_train, X_test, y_test):
    # cross validation
    num_classes = 3
    cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    metric = MulticlassConfusionMatrix(num_classes=num_classes)

    # train a model and check using cross validation
    print("XGBoost using the raw data")
    xgb_model = XGBClassifier(eta=0.1, gamma=10, subsample=0.5, objective="multi:softmax", sampling_method='uniform')
    xgb_model.fit(X_train, y_train)
    print("Cross validation scores:")
    print(cross_val_score(XGBClassifier(), X_train, y_train, cv=cv))

    # run prediction
    y_pred = xgb_model.predict(X_test)

    cm = metric(torch.tensor(y_pred), torch.tensor(y_test))

    correct = 0
    correct += (y_pred == y_test).sum()
    correct /= y_test.shape[0]
    print(f"(XGBoost) Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n")
    print(cm)


def run_randomforest_classifer(X_train, y_train, X_test, y_test):
    num_classes = 3
    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    metric = MulticlassConfusionMatrix(num_classes=num_classes)

    # train a random forest model
    print("Random Forest using the raw data")
    rfc_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=10)
    rfc_model.fit(X_train, y_train)
    print("Cross validation scores:")
    print(cross_val_score(RandomForestClassifier(max_depth=5, random_state=10), X_train, y_train, cv=cv))

    # run prediction
    y_pred = rfc_model.predict(X_test)

    cm = metric(torch.tensor(y_pred), torch.tensor(y_test))

    correct = 0
    correct += (y_pred == y_test).sum().item()
    correct /= y_test.shape[0]
    print(f"(RandomForest) Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n")
    print(cm)

    conf_bad_good = np.where(np.logical_and(y_test == 2, y_pred == 0))
    print(conf_bad_good)


if __name__ == "__main__":
    # for sid in range(13):
    #     if sid == 0 or sid == 9:
    #         continue
    #     print(f"subject id: {sid}")
    #     X_tr, y_tr, X_te, y_te = load_train_test_data(sub_id=sid)
    #     run_xgboost_classifier(X_tr, y_tr, X_te, y_te)
    #     run_randomforest_classifer(X_tr, y_tr, X_te, y_te)

    X_tr, y_tr, X_te, y_te = load_train_test_data(sub_id=0)

    # run_xgboost_classifier(X_tr, y_tr, X_te, y_te)
    # run_randomforest_classifer(X_tr, y_tr, X_te, y_te)
    run_mlp(epochs=1000, acc_only=True)
    # run_cnn2d(epochs=1000, arch='cnn2d')
    # run_cnn2d(epochs=1000, arch='my_resnet', multichannel=True, acc_only=True)
    # run_cnn2d(epochs=1000, arch='resnet18_pretrained', multichannel=True)
