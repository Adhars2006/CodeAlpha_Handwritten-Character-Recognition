import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the CNN model (same as in model.py)
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

# Load the model
num_classes = 47
model = CNN(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Classes
classes = [str(i) for i in range(10)] + [chr(65+i) for i in range(26)] + ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# Load a few test images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.EMNIST(root='./dataset', split='balanced', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Get first batch
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Predict
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

print("Sample Predictions:")
for i in range(10):
    actual = classes[labels[i].item()]
    pred = classes[predicted[i].item()]
    print(f"Image {i+1}: Actual: {actual}, Predicted: {pred}")