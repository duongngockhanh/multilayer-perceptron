import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

start_time = time.time()
##### START


data_path = "Auto_MPG_data.csv"
df = pd.read_csv(data_path)
X_np = df.drop(columns=["MPG"]).values
y_np = df[["MPG"]].values
X_torch = torch.tensor(X_np, dtype=torch.float32)
y_torch = torch.tensor(y_np, dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(X_torch, y_torch, test_size=0.3, random_state=1234, shuffle=True)

_MEAN = X_train.mean(axis=0)
_STD = X_train.std(axis=0)

X_train = (X_train-_MEAN)/_STD
X_val = (X_val-_MEAN)/_STD

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(9, 1)
    def forward(self, x):
        out = self.linear(x)
        return out

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def r_squared(y_pred, y_true):
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    mean_true = torch.mean(y_true)
    ss_residual = torch.sum((y_pred - y_true) ** 2)
    ss_total = torch.sum((y_true - mean_true) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2

num_epochs = 100
train_losses = []
val_losses = []
train_r2 = []
val_r2 = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_labels = []
    train_pred = []
    for samples, labels in train_loader:
        optimizer.zero_grad()
        samples, labels = samples.to(device), labels.to(device)
        y_pred = model(samples)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_labels.extend(labels.tolist())
        train_pred.extend(y_pred.tolist())
    
    train_losses.append(train_loss/len(train_loader))
    train_r2.append(r_squared(train_pred, train_labels))


    model.eval()
    val_loss = 0.0
    val_labels = []
    val_pred = []
    with torch.no_grad():
        for samples, labels in val_loader:
            optimizer.zero_grad()
            samples, labels = samples.to(device), labels.to(device)
            y_pred = model(samples)
            loss = criterion(y_pred, labels)

            val_loss += loss.item()
            val_labels.extend(labels.tolist())
            val_pred.extend(y_pred.tolist())

    val_losses.append(val_loss)
    val_r2.append(r_squared(val_pred, val_labels))

    print(f"Epoch {epoch+1}/{num_epochs}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")


fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
ax[0].plot(train_losses, label="train_losses")
ax[0].plot(val_losses, label="val_losses")
ax[0].set(
    xlabel="Epoch",
    ylabel="Loss",
    title="Train Val Loss"
)

ax[1].plot(train_r2, label="train_r2")
ax[1].plot(val_r2, label="val_r2")
ax[1].set(
    xlabel="Epoch",
    ylabel="R2",
    title="Train Val R2"
)

fig.savefig("c1_visualization.png")


##### END
end_time = time.time()

print(f"Total time: {round(end_time - start_time, 2)}s")
