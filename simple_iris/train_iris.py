import torch
import torch.nn as nn
import torch.optim as optim
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
normalizer = StandardScaler()
normalizer.fit_transform(X_train)
normalizer.transform(X_test)

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
for epoch in num_epochs:
    epoch_loss = []
    for x, y in zip(X_train, y_train):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
    losses.append(sum(epoch_loss)/len(epoch_loss))
