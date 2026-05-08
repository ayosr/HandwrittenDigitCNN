import streamlit as st
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from model import DigitCNN
from utils import preprocess_uploaded_image

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    model.eval()
    return model, device


def main():
    st.title("Handwritten Digit Recognition")
    st.write("Upload or take a picture of a single handwritten digit.")

    model, device = load_model()

    input_method = st.radio(
        "Choose image input method:",
        ["Upload Image", "Take Picture"]
    )

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    else:
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = Image.open(camera_file)

    if image is not None:
        st.subheader("Original Image")
        st.image(image, caption="Input image", width=250)

        processed_tensor = preprocess_uploaded_image(image).to(device)

        # Show processed image for debugging
        processed_img = processed_tensor.squeeze().cpu().numpy()
        st.subheader("Processed 28x28 Image")
        st.image(processed_img, caption="Model input", width=150, clamp=True)

        with torch.no_grad():
            outputs = model(processed_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_digit].item() * 100

        st.subheader("Prediction Result")
        st.success(f"Predicted Digit: {predicted_digit}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")
        probs = probabilities[0].cpu().numpy()
        for i in range(10):
            st.write(f"Digit {i}: {probs[i] * 100:.2f}%")


if __name__ == "__main__":
    main()