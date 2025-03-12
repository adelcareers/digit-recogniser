import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64):
    """
    Load the MNIST dataset and return DataLoaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean & std
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="../data", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="../data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_mnist()
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(test_loader.dataset)} test samples.")


# show sample of the dataset


import matplotlib.pyplot as plt

# Load the dataset
train_loader, test_loader = load_mnist()

# Get a batch of training data
images, labels = next(iter(train_loader))

# Display a few sample images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")

plt.show()
