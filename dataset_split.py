import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def collect_images_by_class(source_path):
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")
    
    images_by_class = defaultdict(list)
    
    for item in source_path.iterdir():
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
        raise ValueError("No classes with .jpg images found in source directory")
    
    total_images = sum(len(images) for images in images_by_class.values())
    print(f"Total .jpg images found: {total_images}")
    
    return images_by_class

def create_output_directories(source_path, images_by_class):
    """
    Create train and validation directories with class subfolders.
    
    Args:
        source_path (Path): Path to the source directory
        images_by_class (dict): Dictionary of images by class
        
    Returns:
        tuple: Paths to train and validation directories
    """
    parent_dir = source_path.parent
    train_path = parent_dir / "train"
    val_path = parent_dir / "validation"
    
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    for class_name in images_by_class.keys():
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    
    return train_path, val_path

def split_and_distribute_images(images_by_class, train_path, val_path, train_ratio=0.9):
    """
    Args:
        images_by_class (dict): Dictionary of images by class
        train_path (Path): Path to training directory
        val_path (Path): Path to validation directory
        train_ratio (float): Ratio of images for training (default: 0.9 for 90%)
    """
    total_train = 0
    total_val = 0
    
    print(f"\nSplitting images with {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% validation ratio:")
    
    for class_name, images in images_by_class.items():
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        
        total_images = len(shuffled_images)
        train_count = int(total_images * train_ratio)
        val_count = total_images - train_count
        
        train_images = shuffled_images[:train_count]
        val_images = shuffled_images[train_count:]
        
        print(f"  {class_name}: {train_count} train, {val_count} validation (of {total_images} total)")
        
        train_copied = 0
        for image_path in train_images:
            dst_path = train_path / class_name / image_path.name
            try:
                shutil.copy2(image_path, dst_path)
                train_copied += 1
            except Exception as e:
                print(f"Error copying {image_path} to train: {e}")
        
        val_copied = 0
        for image_path in val_images:
            dst_path = val_path / class_name / image_path.name
            try:
                shutil.copy2(image_path, dst_path)
                val_copied += 1
            except Exception as e:
                print(f"    Error copying {image_path} to validation: {e}")
        
        total_train += train_copied
        total_val += val_copied
    
    print(f"\nDistribution complete:")
    print(f"  - Training: {total_train} images")
    print(f"  - Validation: {total_val} images")
    print(f"  - Total: {total_train + total_val} images")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Split .jpg images into train/validation sets with class folders')
    parser.add_argument('source_directory', help='Path to the source directory containing class folders with .jpg images')
    parser.add_argument('--train-ratio', type=float, default=0.9, 
                       help='Ratio of images for training (default: 0.9 for 90%%)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    if not 0 < args.train_ratio < 1:
        print("Error: train-ratio must be between 0 and 1")
        return 1
    
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    try:
        source_path = Path(args.source_directory).resolve()
        print(f"Processing source directory: {source_path}")

        images_by_class = collect_images_by_class(source_path)
        train_path, val_path = create_output_directories(source_path, images_by_class)
        split_and_distribute_images(images_by_class, train_path, val_path, args.train_ratio)
        
        print(f"\nSuccess! Created train/validation split:")
        print(f"  - Training: {train_path}")
        print(f"  - Validation: {val_path}")
        print(f"Each directory contains class subfolders with the split images.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

