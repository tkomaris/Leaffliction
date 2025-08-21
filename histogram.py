import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from transformations import gray_scale


def histogram(img: Image.Image):
    np_img = np.asarray(img)
    image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(gray_scale(img), cv2.MORPH_CLOSE, kernel)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    lab = cv2.cvtColor(masked_image, cv2.COLOR_RGB2LAB)  # LAB color space
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)  # HSV color space

    channels = {
        "Blue": (masked_image[:, :, 2], "blue"),
        "Blue-Yellow": (lab[:, :, 2], "yellow"),
        "Green": (masked_image[:, :, 1], "green"),
        "Green-Magenta": (lab[:, :, 1], "magenta"),
        "Hue": (hsv[:, :, 0], "purple"),
        "Lightness": (lab[:, :, 0], "grey"),
        "Red": (masked_image[:, :, 0], "red"),
        "Saturation": (hsv[:, :, 1], "cyan"),
        "Value": (hsv[:, :, 2], "orange"),
    }

    figure = plt.figure(figsize=(10, 6))
    for channel_name, (channel_data, color) in channels.items():
        non_masked_values = channel_data[mask > 0]
        hist, _ = np.histogram(non_masked_values, bins=256, range=(0, 256))
        total_pixels = hist.sum()
        hist_percentage = (hist / total_pixels) * 100
        plt.plot(hist_percentage, color=color, label=channel_name)

    plt.title("Histogram for Multiple Color Channels")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Portion of pixels (%)")
    plt.legend(loc="upper right", fontsize=8)
    plt.xlim([0, 255])
    plt.grid(True, linestyle="--", alpha=0.5)

    return figure
