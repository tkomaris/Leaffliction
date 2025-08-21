import argparse
from PIL import Image
import matplotlib.pyplot as plt
from os import makedirs, scandir
from os.path import isfile, isdir, join

from histogram import histogram
from transformations import transforms


def manipulateImage(img: Image.Image, transform: str):
    res = []
    if transform:
        name, func = transforms[transform]
        res.append((name, func(img)))
    else:
        for (name, func) in transforms.values():
            res.append((name, func(img)))
    return res


def showSingleImageTransforms(path: str, transform: str):
    with Image.open(path) as file:
        file.load()
        if transform != "histogram":
            images = [("Original", file)] + manipulateImage(file, transform)

            _, axs = plt.subplots(1, len(images), figsize=(3 * len(images), 4))
            if len(images) == 1:
                axs = [axs]
            for ax, (title, img) in zip(axs, images):
                ax.set_title(title, fontsize=12)
                cmap = "gray" if title == "Gaussian blur" else None
                ax.imshow(img, cmap=cmap)
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        if not transform or transform == "histogram":
            histogram(file)
            plt.show()


def saveFileTransforms(filepath: str, dst: str, transform: str):
    with Image.open(filepath) as file:
        file.load()
        if transform != "histogram":
            images = [("Original", file)] + manipulateImage(file, transform)
            for (title, img) in images:
                filename = f"{dst}/{filepath.split('/')[-1][:-4]}_{title}.jpg"
                plt.figure()
                plt.imshow(img)
                plt.savefig(filename)
                plt.close()
        if not transform or transform == "histogram":
            filename = f"{dst}/{filepath.split('/')[-1][:-4]}_Histogram.jpg"
            histogram_figure = histogram(file)
            histogram_figure.savefig(filename)
            plt.close(histogram_figure)


def createTransforms(src: str, dst: str, transform: str):
    makedirs(dst, exist_ok=True)
    filenames = [entry.name for entry in scandir(src) if entry.is_file()]
    for filename in filenames:
        saveFileTransforms(join(src, filename), dst, transform)


if __name__ == '__main__':
    valid_tfs = list(transforms.keys()) + ["histogram"]
    parser = argparse.ArgumentParser(description='Describes the data')
    parser.add_argument("-src", required=True,
                        help="Source directory containing images")
    parser.add_argument("-dst", help="Destination directory for output images")
    parser.add_argument("-tf", required=False, choices=valid_tfs,
                        help="Select transform:\
                        'g_blur' - gaussian blur,\
                        'mask' - mask,\
                        'roi' - ROI objects,\
                        'analyze' - analyze objects,\
                        'landmarks' - pseudolandmarks,\
                        'histogram' - channels histogram")
    args = parser.parse_args()
    if isfile(args.src):
        showSingleImageTransforms(args.src, args.tf)
    elif isdir(args.src):
        createTransforms(args.src, args.dst, args.tf)
    else:
        print("Error: wrong path")


def transformation_task(image_path: str, show_original: bool):
    with Image.open(image_path) as img:
        img.load()
        # Apply a default transformation, e.g., gaussian blur
        # You might want to make this configurable or based on user input
        transformed_img_array = transforms["g_blur"][1](img)
        return [None, None, None, transformed_img_array]
