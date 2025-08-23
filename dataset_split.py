import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys


def reset_dirs(paths: list[Path]):
    for path in paths:
        if path.exists():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

        path.mkdir(parents=True)


def collect_images_by_class(src_path):
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_path}")

    images_by_class = defaultdict(list)

    for item in src_path.iterdir():
        if item.is_dir():
            class_name = item.name
            jpg_files = []

            for file_path in item.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.jpg':
                    jpg_files.append(file_path)

            if jpg_files:
                images_by_class[class_name] = jpg_files
                print(f"Class '{class_name}': {len(jpg_files)} .jpg images")
            else:
                print(f"Warning: No .jpg images found in class '{class_name}'")

    if not images_by_class:
        raise ValueError("No classes with images found in source directory")

    total_images = sum(len(images) for images in images_by_class.values())
    print(f"Total .jpg images found: {total_images}")

    return images_by_class


def create_output_directories(source_path, images_by_class):
    """
    Create train and test directories with class subfolders.

    Args:
        source_path (Path): Path to the source directory
        images_by_class (dict): Dictionary of images by class

    Returns:
        tuple: Paths to train and test directories
    """
    parent_dir = source_path.parent
    submission_path = parent_dir / "submission"
    train_path = submission_path / "augmented_directory"
    test_path = submission_path / "test"

    reset_dirs([submission_path, train_path, test_path])

    for class_name in images_by_class.keys():
        (train_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

    print("Created output directories:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

    return train_path, test_path


def split_and_distribute_images(images_by_class, train_path,
                                test_path, train_ratio=0.9):
    """
    Args:
        images_by_class (dict): Dictionary of images by class
        train_path (Path): Path to training directory
        test_path (Path): Path to test directory
        train_ratio (float): Ratio of images for training (default: 0.9)
    """
    total_train = 0
    total_test = 0

    print(f"\nSplitting images with {train_ratio*100:.0f}% " +
          f"train / {(1-train_ratio)*100:.0f}% test ratio:")

    for class_name, images in images_by_class.items():
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        total_images = len(shuffled_images)
        train_count = int(total_images * train_ratio)
        test_count = total_images - train_count

        train_images = shuffled_images[:train_count]
        test_images = shuffled_images[train_count:]

        print(f"  {class_name}: {train_count} train, " +
              f"{test_count} test (of {total_images} total)")

        train_copied = 0
        for image_path in train_images:
            dst_path = train_path / class_name / image_path.name
            try:
                shutil.copy2(image_path, dst_path)
                train_copied += 1
            except Exception as e:
                print(f"Error copying {image_path} to train: {e}")

        test_copied = 0
        for image_path in test_images:
            dst_path = test_path / class_name / image_path.name
            try:
                shutil.copy2(image_path, dst_path)
                test_copied += 1
            except Exception as e:
                print(f"    Error copying {image_path} to test: {e}")

        total_train += train_copied
        total_test += test_copied

    print("\nDistribution complete:")
    print(f"  - Training: {total_train} images")
    print(f"  - Test: {total_test} images")
    print(f"  - Total: {total_train + total_test} images")


def split_dataset(path: str, train_ratio: float):
    source_path = Path(path).resolve()
    print(f"Processing source directory: {source_path}")

    images_by_class = collect_images_by_class(source_path)
    train_path, test_path = create_output_directories(source_path,
                                                      images_by_class)
    split_and_distribute_images(images_by_class, train_path,
                                test_path, train_ratio)

    print("\nSuccess! Created train/test split:")
    print(f"  - Training: {train_path}")
    print(f"  - Test: {test_path}")
    print("Each directory contains class subfolders with the split images.")
    return train_path, test_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Split dataset into train/test sets with class folders')
    parser.add_argument('source_directory',
                        help='Path to the source directory containing\
                              class folders with .jpg images')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of images for training (default: \
                            0.9 for 90%%)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducible results')

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        print("Error: train-ratio must be between 0 and 1")
        return 1

    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    try:
        split_dataset(args.source_directory, args.train_ratio)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
