import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def r_squared(y_pred, y_true):
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    mean_true = torch.mean(y_true)
    ss_residual = torch.sum((y_pred - y_true) ** 2)
    ss_total = torch.sum((y_true - mean_true) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2


def visualize(train_losses, val_losses, train_r2, val_r2):
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

    fig.savefig("visualization.png")