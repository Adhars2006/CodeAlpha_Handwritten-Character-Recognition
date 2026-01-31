import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

# Step 1: Load the EMNIST dataset using torchvision
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./dataset', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./dataset', split='balanced', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# EMNIST classes: 0-9 digits, 10-35 uppercase A-Z, 36-46 some lowercase
num_classes = 47
classes = [str(i) for i in range(10)] + [chr(65+i) for i in range(26)] + ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']  # EMNIST balanced classes

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

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

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
