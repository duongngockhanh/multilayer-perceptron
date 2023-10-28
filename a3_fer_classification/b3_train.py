import time, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torchvision.io import read_image
start_time = time.time()

##### START
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)

batch_size = 256
image_height = 180
image_width = 180
train_dir = "FER-2013/train"
val_dir = "FER-2013/test"

class ImageDataset(Dataset):
    def __init__(self, img_dir, norm):
        self.img_dir = img_dir
        self.norm = norm
        self.resize = Resize((image_height, image_width))
        self.classes = os.listdir(img_dir)
        self.image_files = [(os.path.join(img_dir, cls, img_path), cls) for cls in self.classes for img_path in os.listdir(os.path.join(img_dir, cls))]
        self.class_to_idx = {cls:idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx:cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, cls = self.image_files[idx]
        image = self.resize(read_image(img_path))
        image = image.type(torch.float32)
        label = self.class_to_idx[cls]
        if self.norm:
            image = image / 127.5 - 1
        return image, label
    

train_dataset = ImageDataset(train_dir, False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = ImageDataset(val_dir, False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, input_dims, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dims, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, num_classes)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        out = self.linear4(x)
        return out
    
model = Model(input_dims=image_height*image_width, num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.006)

num_epochs = 2
train_losses = []
val_losses = []
train_acc = []
val_acc = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0
    train_count = 0
    for samples, labels in train_loader:
        optimizer.zero_grad()
        samples, labels = samples.to(device), labels.to(device)
        output = model(samples)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += (torch.argmax(output, axis=1) == labels).sum().item()
        train_count += len(labels)
    train_losses.append(train_loss/len(train_loader))
    train_acc.append(train_accuracy/train_count)

    model.eval()
    val_loss = 0.0
    val_accuracy = 0
    val_count = 0
    with torch.no_grad():
        for samples, labels in val_loader:
            samples, labels = samples.to(device), labels.to(device)
            output = model(samples)
            loss = criterion(output, labels)

            val_loss += loss.item()
            val_accuracy += (torch.argmax(output, axis=1) == labels).sum().item()
            val_count += len(labels)
        val_losses.append(val_loss/len(val_loader))
        val_acc.append(val_accuracy/val_count)

    print(f"Epoch {epoch+1}/{num_epochs}: Train_loss: {train_loss:.4f}, Train_acc: {train_acc[-1]:.4f}, Val_loss: {val_loss:.4f}, Val_acc: {val_acc[-1]:.4f}")


fig = plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train_loss', color='green')
plt.plot(val_losses, label='val_loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='train_acc', color='green')
plt.plot(val_acc, label='val_acc', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

fig.savefig("c3_visualization.png")            

##### END
end_time = time.time()

print(f"Total time: {round(end_time - start_time, 2)}s")