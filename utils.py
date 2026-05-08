from PIL import Image, ImageOps
import numpy as np
import torch


def preprocess_uploaded_image(image: Image.Image) -> torch.Tensor:
    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy
    img_array = np.array(image).astype("float32")

    # Invert so dark writing on light paper becomes MNIST-like
    img_array = 255.0 - img_array

    # Normalize
    img_array /= 255.0

    # Reshape to (1, 1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=0)

    return torch.tensor(img_array, dtype=torch.float32)