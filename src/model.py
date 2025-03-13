'''
Plan for the Model: CNN for MNIST


CNN Model (CNN class)

Uses two convolutional layers with ReLU and max pooling.
Ends with two fully connected layers for classification.
Data Loading (load_data)

Uses torchvision.datasets.MNIST to download and load data.
Training Loop (train_model)

Runs multiple epochs.
Uses Adam optimizer and CrossEntropyLoss.
Testing Loop (test_model)

Evaluates accuracy on the test dataset.
Model Saving

Saves the trained model in ../models/mnist_cnn.pth.
Loss Visualization

Plots the training loss over epochs.

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Check if MPS (Apple Metal GPU) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")  # Will print 'mps' if using GPU, otherwise 'cpu'

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Corrected
        # self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        
        # Print only for the first batch
        if not hasattr(self, 'printed'):
            print("Shape before flattening:", x.shape)
            x_flattened = x.view(x.size(0), -1)  # Flatten the tensor
            print("Shape after flattening:", x_flattened.shape)
            self.printed = True  # Prevent further prints
        else:
            x_flattened = x.view(x.size(0), -1)

        x = self.relu3(self.fc1(x_flattened))
        x = self.fc2(x)
        return x


# Function to load MNIST dataset
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return train_losses

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Main function
if __name__ == "__main__":
    # Load data
    train_loader, test_loader = load_data()

    # Initialize model, loss function, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training the model...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=5)

    # Test model
    print("Testing the model...")
    test_accuracy = test_model(model, test_loader)

    # Save trained model
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "mnist_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Plot training loss
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()