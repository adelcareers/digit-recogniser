import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Load the trained model
# @st.cache_resource
# def load_model():
#     model = torch.load("../../models/mnist_cnn.pth", map_location=torch.device("cpu"))
#     model.eval()
#     return model

# import os

# @st.cache_resource
# def load_model():
#     model_path = os.path.join(os.path.dirname(__file__), "../../models/mnist_cnn.pth")
#     model_path = os.path.abspath(model_path)  # Convert to absolute path
#     print(f"Loading model from: {model_path}")  # Debugging statement

#     model = torch.load(model_path, map_location=torch.device("cpu"))
#     model.eval()
#     return model

import os
import torch
import torch.nn as nn

# Import the CNN model from model.py
# from src.model import CNN  

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import the CNN model
from src.model import CNN


# @st.cache_resource
# def load_model():
#     model_path = os.path.join(os.path.dirname(__file__), "../../models/mnist_cnn.pth")
#     model_path = os.path.abspath(model_path)  # Convert to absolute path
#     print(f"Loading model from: {model_path}")  # Debugging statement

#     model = CNN()  # Initialize the model architecture
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights
#     model.eval()  # Set model to evaluation mode
#     return model

import sys
import os
import torch
import torch.nn as nn

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import the CNN model
from src.model import CNN

@st.cache_resource
def load_model():
    # Detect if MPS (Mac GPU) is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get absolute model path
    model_path = os.path.join(os.path.dirname(__file__), "../../models/mnist_cnn.pth")
    model_path = os.path.abspath(model_path)  # Convert to absolute path
    print(f"Loading model from: {model_path}")

    # Initialize model
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights onto the detected device
    model.to(device)  # Move model to MPS if available
    model.eval()
    
    return model, device  # Return both the model and device

# Load the model
model, device = load_model()


# model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel image
    transforms.Resize((28, 28)),  # Resize to MNIST format
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize like MNIST dataset
])

# def preprocess_image(image):
#     """Preprocess user-drawn image before prediction."""
#     image = ImageOps.invert(image.convert("L"))  # Invert colors (white digit on black background)
#     image = transform(image)  # Apply transformations
#     image = image.unsqueeze(0)  # Add batch dimension
#     return image

# def predict_digit(image_tensor):
#     """Predict the digit from the processed image."""
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
#         confidence, predicted_label = torch.max(probabilities, 1)  # Get highest confidence prediction
#     return predicted_label.item(), confidence.item()

# def preprocess_image(image):
#     """Preprocess user-drawn image before prediction."""
#     image = ImageOps.invert(image.convert("L"))  # Invert colors (white digit on black background)
#     image = transform(image)  # Apply transformations
#     image = image.unsqueeze(0)  # Add batch dimension
#     image = image.to(device)  # Move to the same device as the model
#     return image
def preprocess_image(image):
    """Preprocess user-drawn image before prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (MNIST has white digits on black background)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize same as training
    ])
    
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move to same device as model

    return image


def predict_digit(image_tensor):
    """Predict the digit from the processed image."""
    image_tensor = image_tensor.to(device)  # Move input to the same device as the model
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
        confidence, predicted_label = torch.max(probabilities, 1)  # Get highest confidence prediction
    return predicted_label.item(), confidence.item()


# Streamlit UI
st.title("MNIST Digit Recognizer")
st.write("Draw a digit below and click 'Predict'.")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Process and predict digit when user clicks "Predict"
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to PIL Image
        img = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
        
        # Preprocess image and make prediction
        img_tensor = preprocess_image(img)
        predicted_digit, confidence = predict_digit(img_tensor)

        # Display results
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # User feedback
        true_label = st.text_input("Enter the correct digit (optional for feedback):", "")
        if st.button("Submit Feedback"):
            st.write("Thank you for your feedback!")

st.write("👆 Draw a number (0-9) and click 'Predict' to see the result.")