# Handwritten Digit Recognition System using PyTorch CNN

## Project Description
This project is a handwritten digit recognition system built using a Convolutional Neural Network (CNN) 
trained on the MNIST dataset. It includes a web-based GUI created with Streamlit that allows users to 
upload their own handwritten digit images and receive a predicted digit output from 0 to 9.

The uploaded image is preprocessed by converting it to grayscale, resizing it to 28×28 pixels, 
normalizing it, and then passing it through the trained CNN model for prediction.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib

## Dataset
- MNIST Handwritten Digit Dataset

## Features
- Trains a CNN on MNIST
- Evaluates model accuracy and loss
- Uploads `.png`, `.jpg`, `.jpeg` files
- Displays uploaded image
- Predicts the handwritten digit
- Shows prediction confidence

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python train_model.py

### 3. Run the GUI

streamlit run app.py

## Screenshot(s)
<img width="1436" height="781" alt="image" src="https://github.com/user-attachments/assets/cd772299-6da3-4cd1-b0c9-d23fd691c877" />
