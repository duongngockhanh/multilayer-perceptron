import time
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import Model1
from utils import *

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

train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model1().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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


visualize(train_losses, val_losses, train_r2, val_r2)

##### END
end_time = time.time()

print(f"Total time: {round(end_time - start_time, 2)}s")
