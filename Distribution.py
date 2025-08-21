import argparse
from os import walk
from os.path import isdir
import matplotlib.pyplot as plt

color_by_type = {
    "healthy": "#4CAF50",
    "rot": "#4B3621",
    "scab": "#8B4513",
    "rust": "#B7410E",
    "Esca": "#8B4513",
    "spot": "#B7410E",
    }

IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")

def listFolder(path):
    dirs = []
    for (dirpath, _, filenames) in walk(path):
        img_files = [f for f in filenames if f.lower().endswith(IMG_EXT)]
        if (len(img_files) > 0):
            dirs.append({"path": dirpath, "filenames": img_files})
    return dirs


def getColors(dirs: list):
    try:
        colors = []
        for dir in dirs:
            colors.append(color_by_type[dir["type"].split("_")[-1]])
        return colors
    except KeyError:
        return None


def analyzeDataset(path: str, verbose=False):
    dirs = listFolder(path)
    labels = []
    filenames = []
    if len(dirs) == 0:
        print("Empty or non-existing folder")
        return
    for dir in dirs:
        labels.append(dir["path"].split('/')[-1])
        filenames.append(dir["filenames"])
    if verbose:
        colors = getColors(dirs)
        sizes = list(map(len, filenames))
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
        fig.suptitle(path.split("/")[-1] + " class distributions")
        axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        axs[1].bar(labels, sizes, label=labels, color=colors, zorder=2)
        axs[1].grid(axis="y", zorder=1)

        plt.show()
    return labels, filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describes the data')
    parser.add_argument("path", type=str, help='add a folder path')
    args = parser.parse_args()
    if isdir(args.path):
        analyzeDataset(args.path, True)
    else:
        print("Passed path is not a folder or doesn't exist")
