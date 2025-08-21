import argparse
from Distribution import listFolder
from PIL import Image
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isfile, isdir, join, splitext
from shutil import copytree, copy2
from augmentations import auguments


def manipulateImage(img: Image.Image):
    res = []
    for augument in auguments.values():
        res.append(augument(img))
    return res


def get_filename(path: str, suffix: str):
    name, ext = splitext(path)
    return f"{name}_{suffix}{ext}"


def singleImageAuguments(path: str):
    with Image.open(path) as file:
        file.load()
        images = [file] + manipulateImage(file)

        _, axs = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
        if len(images) == 1:
            axs = [axs]
        titles = ["Original"] + list(auguments.keys())
        for ax, img, title in zip(axs, images, titles):
            ax.set_title(title, fontsize=12)
            ax.imshow(img)
            ax.axis('off')
            if title != "Original":
                img.save(get_filename(path, title))
        plt.tight_layout()
        plt.show()


def addFileAugument(path: str, i: int):
    augument_index = i % len(auguments)
    with Image.open(path) as file:
        file.load()
        new_image = list(auguments.values())[augument_index](file)
        suffix = list(auguments.keys())[augument_index]
        new_image.save(get_filename(path, suffix))


def enrichDataset(path: str):
    dirs = listFolder(path)
    sizes = list(map(len, [dir["filenames"] for dir in dirs]))
    maxSize = max(sizes)
    for dir in dirs:
        filenames = dir["filenames"]
        type_path = dir["path"]
        if len(filenames) < maxSize:
            diff = maxSize - len(filenames)
            while diff > 0:
                filename = filenames[diff % len(filenames)]
                file_path = join(type_path, filename)
                addFileAugument(file_path, diff)
                diff -= 1

    dirs = listFolder(path)
    sizes = list(map(len, [dir["filenames"] for dir in dirs]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augments the data set')
    parser.add_argument(
        '-s', '--src',
        type=str,
        required=True,
        help='add a source file/folder'
    )
    parser.add_argument(
        '-d', '--dst',
        type=str,
        required=True,
        help='add a destination folder'
    )
    args = parser.parse_args()
    makedirs(args.dst, exist_ok=True)
    if not isdir(args.dst):
        print("Destination is not a folder")
    elif isfile(args.src):
        new_file_path = copy2(args.src, args.dst)
        singleImageAuguments(new_file_path)
    elif isdir(args.src):
        copytree(args.src, args.dst, dirs_exist_ok=True)
        enrichDataset(args.dst)
    else:
        print("Source file reading error")
