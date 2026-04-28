import streamlit as st
import torch
from PIL import Image
import torch.nn.functional as F
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
    st.write("Upload an image of a handwritten digit (0–9), and the model will predict it.")

    model, device = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded digit image", width=200)

        processed_image = preprocess_uploaded_image(image).to(device)

        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_digit].item() * 100

        st.subheader("Prediction Result")
        st.success(f"Predicted Digit: {predicted_digit}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")
        for i in range(10):
            st.write(f"Digit {i}: {probabilities[0][i].item() * 100:.2f}%")


if __name__ == "__main__":
    main()