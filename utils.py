from PIL import Image
import numpy as np
import torch


def preprocess_uploaded_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image).astype("float32")

    # Invert colors to match MNIST style: white digit on black background
    img_array = 255.0 - img_array

    # Normalize to [0, 1]
    img_array /= 255.0

    # Shape: (1, 1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=0)

    tensor = torch.tensor(img_array, dtype=torch.float32)
    return tensor