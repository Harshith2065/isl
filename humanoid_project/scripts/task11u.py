import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 480))
    return image_resized


if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")
    try:
        processed_image = load_and_preprocess_image(image_path)
        plt.imshow(processed_image)
        plt.title("Preprocessed Image (640x480 RGB)")
        plt.axis("off")
        plt.show()

    except ValueError as e:
        print(e)
