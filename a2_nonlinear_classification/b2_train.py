import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


start_time = time.time()
##### START
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = "NonLinear_data.npy"
data = np.load(data_path, allow_pickle=True).item()
X_np, y_np = data['X'], data['labels']
X_torch, y_torch = torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.long)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_torch, y_torch)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SoftMaxRegression(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(SoftMaxRegression, self).__init__()
        self.linear1 = nn.Linear(input_dims, 128)
        self.linear2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        out = self.relu(x)
        return out

model = SoftMaxRegression(input_dims=2, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 500
train_losses = []
train_acc = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    accuracy = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accuracy += (torch.argmax(outputs, 1) == y).sum().item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    accuracy /= y_np.size
    train_acc.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {train_loss:.4f},  Train_Acc: {accuracy:.4f}")


fig = plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Error [CrosEntr]')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)

fig.savefig("c2_visualization.png")

##### END
end_time = time.time()

print(f"Total time: {round(end_time - start_time, 2)}s")