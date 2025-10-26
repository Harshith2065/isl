"""Task 1.1: Load and preprocess an image for pose estimation."""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 480))
    return image_resized


if __name__ == "__main__":
    path = "../images/pose1.jpg"  # replace with an actual image file
    img = load_and_preprocess_image(path)
    plt.imshow(img)
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()
