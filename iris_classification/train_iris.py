import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1a. Load dataset
data = load_iris()
X = data.data
y = data.target

# 1b. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1c. Normalize dataset
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

# 1d. Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 2a. Initial model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

# 2b. Initial hyper-parameter
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 20
losses = []
accs = []


# 2c. Train
for epoch in tqdm(range(num_epochs)):
    epoch_loss = []
    for sample, label in zip(X_train, y_train):
        optimizer.zero_grad()
        pred = model(sample)
        loss = criterion(pred, label.long())
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    losses.append(sum(epoch_loss)/len(epoch_loss))

# 3a. Evaluate
with torch.no_grad():
    preds = model(X_test)

y_preds = torch.argmax(preds, dim=1)
acc = sum(y_preds == y_test)/len(y_test)
print(f"Accuracy: {acc.item() * 100}%")

# 3b. Visulize
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.savefig("loss.png")

# 3c. Save weights
torch.save(model.state_dict(), "last.pt")