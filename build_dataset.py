#!/usr/bin/env python3
"""
Build Multitask Dataset Script

This script builds and saves the multitask dataset using the MultitaskDatasetBuilder.
It creates the dataset file that the trainer expects.

Usage:
    python build_multitask_dataset.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.dataset_builder import MultitaskDatasetBuilder

def main():
    """Build and save the multitask dataset."""
    
    print("="*60)
    print("BUILDING MULTITASK DATASET")
    print("="*60)
    
    # Initialize the dataset builder
    builder = MultitaskDatasetBuilder(
        videos_dir="data/raw",
        no_exercise_dir="data/no_exercise", 
        labels_dir="data/labels",
        fps=30,
        window_size=30,
        margin_frames=12, sigma=3.40,
        no_exercise_ratio=0.3
    )
    
    # Build the dataset
    print("\nBuilding dataset...")
    X, y_classification, y_segmentation, metadata = builder.build_dataset()
    
    # Create output directory
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataset
    dataset_path = os.path.join(output_dir, "multitask_dataset.npz")
    print(f"\nSaving dataset to {dataset_path}...")
    
    np.savez(
        dataset_path,
        X=X,
        y_classification=y_classification,
        y_segmentation=y_segmentation,
        metadata=metadata
    )
    
    print(f"\nDataset saved successfully!")
    print(f"Dataset shape: {X.shape}")
    print(f"Classification labels shape: {y_classification.shape}")
    print(f"Segmentation labels shape: {y_segmentation.shape}")
    print(f"Total sequences: {len(X)}")
    
    # Print class distribution
    print(f"\nClass distribution:")
    for exercise_type, count in metadata['class_distribution'].items():
        percentage = count / len(y_classification) * 100
        print(f"  {exercise_type}: {count} sequences ({percentage:.1f}%)")
    
    # Print segmentation statistics
    positive_seg = np.sum(y_segmentation > 0.5)
    total_seg = y_segmentation.size
    print(f"\nSegmentation statistics:")
    print(f"  Positive labels: {positive_seg} / {total_seg} ({positive_seg/total_seg*100:.1f}%)")
    print(f"  Max value: {np.max(y_segmentation):.3f}")
    print(f"  Mean value: {np.mean(y_segmentation):.3f}")
    
    print(f"\nDataset is ready for training!")
    print(f"Use: python src/training/train.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
