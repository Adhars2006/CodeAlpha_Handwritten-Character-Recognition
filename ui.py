import streamlit as st
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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

# Preprocess function
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    image_np = np.array(image)
    # Apply EMNIST rotation: flip left-right, then rotate 90 degrees clockwise
    image_np = np.fliplr(image_np)
    image_np = np.rot90(image_np, k=1, axes=(0, 1))
    # Convert back to PIL
    image = Image.fromarray(image_np)
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

# Streamlit UI
st.title("Handwritten Character Recognition")
st.write("Upload an image of a handwritten character (preferably 28x28 pixels)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]
    
    st.write(f"Predicted Character: {predicted_class}")