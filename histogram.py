import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def histogram(img: Image.Image):
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        colors = ('b', 'g', 'r')
        title = 'Color Histogram'
    else:
        colors = ('k',)
        title = 'Grayscale Histogram'

    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    return plt

