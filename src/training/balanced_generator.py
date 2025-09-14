#!/usr/bin/env python3
"""
Balanced Data Generator for Class Imbalance

Creates batches with balanced representation of positive and negative sequences
to address the severe class imbalance problem (1:52 ratio).
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List


class BalancedSequenceGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator that creates balanced batches for training.
    
    Ensures each batch contains a good mix of sequences with positive labels
    and sequences with only negative labels to combat class imbalance.
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y_cls: np.ndarray, 
                 y_seg: np.ndarray,
                 batch_size: int = 32,
                 positive_ratio: float = 0.4,
                 shuffle: bool = True):
        """
        Initialize the balanced generator.
        
        Args:
            X: Feature sequences (num_samples, window_size, num_features)
            y_cls: Classification labels (num_samples,)
            y_seg: Segmentation labels (num_samples, window_size)
            batch_size: Number of samples per batch
            positive_ratio: Ratio of positive sequences in each batch (0.0-1.0)
            shuffle: Whether to shuffle data between epochs
        """
        self.X = X
        self.y_cls = y_cls
        self.y_seg = y_seg
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.shuffle = shuffle
        
        # Split indices into positive and negative sequences
        self._split_indices()
        
        # Calculate batch composition
        self.positive_per_batch = int(batch_size * positive_ratio)
        self.negative_per_batch = batch_size - self.positive_per_batch
        
        print(f"Balanced Generator initialized:")
        print(f"  Total sequences: {len(X)}")
        print(f"  Positive sequences: {len(self.positive_indices)} ({len(self.positive_indices)/len(X)*100:.1f}%)")
        print(f"  Negative sequences: {len(self.negative_indices)} ({len(self.negative_indices)/len(X)*100:.1f}%)")
        print(f"  Batch composition: {self.positive_per_batch} positive + {self.negative_per_batch} negative")
        
        self.on_epoch_end()
    
    def _split_indices(self):
        """Split sequence indices into positive and negative based on segmentation labels."""
        # A sequence is positive if it contains any positive segmentation labels
        has_positive = np.sum(self.y_seg, axis=1) > 0
        self.positive_indices = np.where(has_positive)[0]
        self.negative_indices = np.where(~has_positive)[0]
    
    def __len__(self):
        """Return number of batches per epoch."""
        # Base on the limiting factor (usually positive sequences)
        max_positive_batches = len(self.positive_indices) // self.positive_per_batch
        max_negative_batches = len(self.negative_indices) // self.negative_per_batch
        return min(max_positive_batches, max_negative_batches)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate one batch of data."""
        # Get indices for this batch
        pos_start = index * self.positive_per_batch
        pos_end = pos_start + self.positive_per_batch
        neg_start = index * self.negative_per_batch
        neg_end = neg_start + self.negative_per_batch
        
        # Sample positive and negative sequences
        pos_batch_indices = self.shuffled_positive_indices[pos_start:pos_end]
        neg_batch_indices = self.shuffled_negative_indices[neg_start:neg_end]
        
        # Combine and shuffle within batch
        batch_indices = np.concatenate([pos_batch_indices, neg_batch_indices])
        if self.shuffle:
            np.random.shuffle(batch_indices)
        
        # Extract batch data
        batch_X = self.X[batch_indices]
        batch_y_cls = self.y_cls[batch_indices]
        batch_y_seg = self.y_seg[batch_indices]
        
        return batch_X, {'classification_output': batch_y_cls, 'segmentation_output': batch_y_seg}
    
    def on_epoch_end(self):
        """Called at the end of each epoch to shuffle data."""
        if self.shuffle:
            self.shuffled_positive_indices = np.random.permutation(self.positive_indices)
            self.shuffled_negative_indices = np.random.permutation(self.negative_indices)
        else:
            self.shuffled_positive_indices = self.positive_indices.copy()
            self.shuffled_negative_indices = self.negative_indices.copy()


def create_balanced_generators(X: np.ndarray, 
                             y_cls: np.ndarray, 
                             y_seg: np.ndarray,
                             validation_split: float = 0.2,
                             batch_size: int = 32,
                             positive_ratio: float = 0.4) -> Tuple[BalancedSequenceGenerator, BalancedSequenceGenerator]:
    """
    Create balanced train and validation generators.
    
    Args:
        X: Feature sequences
        y_cls: Classification labels  
        y_seg: Segmentation labels
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for training
        positive_ratio: Ratio of positive sequences per batch
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Split data
    n_samples = len(X)
    n_val = int(n_samples * validation_split)
    
    # Use random indices for split
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Create train/val splits
    X_train, X_val = X[train_indices], X[val_indices]
    y_cls_train, y_cls_val = y_cls[train_indices], y_cls[val_indices]
    y_seg_train, y_seg_val = y_seg[train_indices], y_seg[val_indices]
    
    # Create generators
    train_gen = BalancedSequenceGenerator(
        X_train, y_cls_train, y_seg_train,
        batch_size=batch_size,
        positive_ratio=positive_ratio,
        shuffle=True
    )
    
    val_gen = BalancedSequenceGenerator(
        X_val, y_cls_val, y_seg_val,
        batch_size=batch_size,
        positive_ratio=positive_ratio,  # Keep same ratio for validation
        shuffle=False
    )
    
    return train_gen, val_gen 