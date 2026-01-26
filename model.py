import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

# Step 1: Download the EMNIST dataset from Kaggle
# Note: You need to have Kaggle API installed and authenticated.
os.system('kaggle datasets download -d crawford/emnist -p ./dataset --unzip')

# We'll use emnist-balanced-train.csv and emnist-balanced-test.csv for balanced dataset (47 classes: 0-9, A-Z, some lowercase)

# Step 2: Load the data
train_path = './dataset/emnist-balanced-train.csv'
test_path = './dataset/emnist-balanced-test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# EMNIST classes: 0-9 digits, 10-35 uppercase A-Z, 36-46 some lowercase
num_classes = 47

# Separate labels and features
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize
y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0

# Reshape to images: 28x28, but EMNIST is transposed/rotated, need to rotate
def reshape_and_rotate(images):
    images = images.reshape(-1, 28, 28)
    images = np.fliplr(images)
    images = np.rot90(images, k=1, axes=(1,2))
    return images.reshape(-1, 1, 28, 28)  # Add channel dim

X_train = reshape_and_rotate(X_train)
X_test = reshape_and_rotate(X_test)

# Step 3: Define Dataset
class EMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, shear=10)
])

train_dataset = EMNISTDataset(X_train, y_train, transform=transform)
test_dataset = EMNISTDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 4: Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Step 5: Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Step 6: Evaluate the model
model.eval()
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = softmax(outputs, dim=1)
        pred = torch.max(outputs, 1)[1]
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
        y_probs.extend(probs.tolist())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# ROC-AUC for multi-class
from sklearn.preprocessing import label_binarize
y_true_bin = label_binarize(y_true, classes=range(num_classes))
y_probs = np.array(y_probs)
roc_auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Note on extension: For full word/sentence recognition, combine with CTC loss and RNN (e.g., CRNN model) using datasets like IAM Handwriting Database.

# Optional: Visualize some predictions
def imshow(img):
    plt.imshow(img.squeeze(), cmap='gray')
    plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

for i in range(4):
    print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
    imshow(images[i])
