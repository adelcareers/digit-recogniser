import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
import os
import sys
import psycopg2
from datetime import datetime
import pandas as pd


# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the CNN model and logging function
from src.model import CNN
from src.db import log_prediction

@st.cache_resource
def load_model():
    """Load the trained model and move it to the correct device."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get absolute model path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/mnist_cnn.pth"))
    print(f"Loading model from: {model_path}")

    # Initialize model and load weights
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# Load the model
model, device = load_model()

# Define image preprocessing (ensures input matches model requirements)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
    transforms.Resize((28, 28)),  # Resize to MNIST format
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean & std
])

def preprocess_image(image):
    """Preprocess user-drawn image before prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (MNIST expects white digits on black background)
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move to the same device as the model
    return image

def predict_digit(image_tensor):
    """Predict the digit from the processed image."""
    image_tensor = image_tensor.to(device)  # Ensure tensor is on the correct device
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

if st.button("Predict"):
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3]):  # Ensure valid input
        # Convert canvas to PIL Image
        img = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
        
        # Preprocess image and make prediction
        img_tensor = preprocess_image(img)
        predicted_digit, confidence = predict_digit(img_tensor)

        # Display results
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # Store predicted digit in session state
        st.session_state["predicted_digit"] = predicted_digit

# User feedback input (remains persistent after clicking submit)
true_label = st.text_input("Enter the correct digit (optional for feedback):", key="true_label_input")

# Log prediction to PostgreSQL when user submits feedback
if st.button("Submit Feedback"):
    if "predicted_digit" in st.session_state:
        true_label = int(true_label) if true_label.isdigit() else None
        log_prediction(st.session_state["predicted_digit"], true_label)
        st.success("✅ Prediction logged successfully!")
    else:
        st.warning("⚠️ Please make a prediction first.")


# Display logged predictions
st.subheader("Recent Predictions")

# Fetch data from PostgreSQL
from src.db import fetch_predictions

predictions_df = fetch_predictions()

# Show table if data exists
if predictions_df is not None and not predictions_df.empty:
    st.dataframe(predictions_df)  # Show predictions in an interactive table
else:
    st.write("No logged predictions yet.")
