#!/usr/bin/env python3
"""
Multitask Dataset Builder for 30 FPS Processing

Processes videos at full 30fps without downsampling.
Creates windowed sequences for both classification and segmentation tasks.
"""

import os
import numpy as np
import cv2
from typing import Tuple, List, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.dataset_builder import FeatureExtractor, LabelAugmenter


class MultitaskDatasetBuilder:
    """
    Builds multitask datasets by processing videos at full 30fps.
    No downsampling - processes every frame directly.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", 
                 fps: int = 30, window_size: int = 30):
        """
        Initialize the multitask dataset builder.
        
        Args:
            videos_dir: Directory containing training videos organized by exercise type
            labels_dir: Directory containing manual labels (CSV files)
            fps: Video frame rate (default: 30)
            window_size: Number of frames per sequence (default: 30, representing 1 second at 30fps)
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.fps = fps
        self.window_size = window_size
        
        # Initialize feature extractor and label augmenter
        self.extractor = FeatureExtractor()
        self.augmenter = LabelAugmenter(fps=fps, margin_sec=0.4)  # Use full fps for augmentation
        
        # Exercise types (including 'no-exercise')
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips', 'no-exercise']
        
        print(f"Multitask dataset builder initialized:")
        print(f"  Window size: {window_size} frames (represents {window_size/fps:.1f} seconds)")
        print(f"  FPS: {fps}")
        print(f"  Exercise types: {self.exercise_types}")
    
    def _extract_video_features(self, video_path: str) -> Dict[int, np.ndarray]:
        """
        Extract features from ALL video frames for windowing.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict mapping frame_index -> feature_array for all frames
        """
        cap = cv2.VideoCapture(video_path)
        features = {}
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract features from every frame
                feat = self.extractor.extract_angles(frame)
                if feat is not None:
                    features[frame_count] = feat
                
                frame_count += 1
        finally:
            cap.release()
        
        return features
    
    
    def _load_labels_csv(self, label_path: str, max_frame_idx: int, exercise_type: str) -> np.ndarray:
        """
        Load segmentation labels from CSV file, with fallback to synthetic labels.
        
        Args:
            label_path: Path to the CSV label file
            max_frame_idx: Maximum frame index in the video
            exercise_type: Type of exercise for synthetic label generation
            
        Returns:
            Binary array where 1 indicates repetition start (indexed by frame number)
        """
        # First try to load from CSV
        if os.path.exists(label_path):
            try:
                import pandas as pd
                df = pd.read_csv(label_path)
                labels = np.zeros(max_frame_idx + 1)
                
                # Check for different possible label formats
                if 'label' in df.columns and 'frame_index' in df.columns:
                    frame_indices = df['frame_index'].values
                    rep_labels = df['label'].values
                    
                    # Use original frame indices (no conversion needed)
                    positive_count = 0
                    for idx, label_val in zip(frame_indices, rep_labels):
                        if label_val == 1.0 and 0 <= idx <= max_frame_idx:
                            labels[idx] = 1.0
                            positive_count += 1
                    
                    # If we found actual positive labels, use them
                    if positive_count > 0:
                        print(f"    Found {positive_count} rep markers in CSV")
                        return labels
                        
            except Exception as e:
                print(f"  Error loading labels from {label_path}: {e}")
        
        # Fallback: return all zeros if no valid labels found
        print(f"  No valid labels found, using all zeros")
        return np.zeros(max_frame_idx + 1)

    
    def _create_windowed_sequences(self, features: Dict[int, np.ndarray], labels: np.ndarray, 
                                 exercise_idx: int) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
        """
        Create windowed sequences using sliding window at 30fps.
        
        For each starting position, creates a sequence of consecutive frames:
        - Window starting at frame 0: [0, 1, 2, 3, ..., 29] (30 frames total)
        - Window starting at frame 1: [1, 2, 3, 4, ..., 30] 
        - Window starting at frame 2: [2, 3, 4, 5, ..., 31]
        - etc.
        
        Args:
            features: Dict mapping frame_index -> feature_array for all frames
            labels: Segmentation labels for each frame
            exercise_idx: Exercise type index for classification
            
        Returns:
            Tuple of (sequences, classification_labels, segmentation_labels)
        """
        max_frame = max(features.keys()) if features else 0
        sequences = []
        cls_labels = []
        seg_labels = []
        
        # Create sliding windows with step size of 1 (every frame)
        for start_frame in range(max_frame - self.window_size + 2):  # +2 to include last possible window
            # Calculate frame indices for this window
            frame_indices = list(range(start_frame, start_frame + self.window_size))
            
            # Check if all frames in this window are available
            if all(idx in features and idx < len(labels) for idx in frame_indices):
                # Extract features for this window
                window_features = []
                window_labels = []
                
                for frame_idx in frame_indices:
                    window_features.append(features[frame_idx])
                    window_labels.append(labels[frame_idx])
                
                # Convert to numpy arrays
                sequence = np.array(window_features)  # Shape: (window_size, num_features)
                seg_label = np.array(window_labels)   # Shape: (window_size,)
                
                sequences.append(sequence)
                cls_labels.append(exercise_idx)
                seg_labels.append(seg_label)
        
        return sequences, cls_labels, seg_labels
    
    def build(self, balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Build the complete multitask dataset with 30 FPS processing.
        
        Args:
            balance_classes: Whether to balance the number of samples per exercise
            
        Returns:
            Tuple of (features, classification_labels, segmentation_labels, metadata)
        """
        print("Building multitask dataset with 30 FPS processing...")
        
        all_sequences = []
        all_cls_labels = []
        all_seg_labels = []
        samples_per_exercise = {}
        
        # Process each exercise type
        for exercise_idx, exercise_type in enumerate(self.exercise_types):
            print(f"\nProcessing {exercise_type}...")
            
            try:
                # Get video and label directories for this exercise
                exercise_video_dir = os.path.join(self.videos_dir, exercise_type)
                exercise_label_dir = os.path.join(self.labels_dir, exercise_type)
                
                if not os.path.exists(exercise_video_dir):
                    print(f"  Directory not found: {exercise_video_dir}")
                    continue
                    
                video_files = [f for f in os.listdir(exercise_video_dir) if f.endswith('.mp4')]
                print(f"  Found {len(video_files)} videos for {exercise_type}")
                
                if not video_files:
                    print(f"  No videos found for {exercise_type}")
                    continue
                
                # Process videos
                exercise_sequences = []
                exercise_cls_labels = []
                exercise_seg_labels = []
                
                for video_file in video_files:
                    video_path = os.path.join(exercise_video_dir, video_file)
                    print(f"    Processing {video_file}...")
                    
                    # Extract features from video (at 30fps)
                    features = self._extract_video_features(video_path)
                    if not features:
                        print(f"    No features extracted from {video_file}")
                        continue
                    
                    # Load corresponding labels (no labels needed for no-exercise)
                    if exercise_type == 'no-exercise':
                        # For no-exercise videos, all segmentation labels are 0
                        max_frame_idx = max(features.keys()) if features else 0
                        seg_labels = np.zeros(max_frame_idx + 1)
                    else:
                        label_file = video_file.replace('.mp4', '.csv')
                        label_path = os.path.join(exercise_label_dir, label_file)
                        max_frame_idx = max(features.keys()) if features else 0
                        seg_labels = self._load_labels_csv(label_path, max_frame_idx, exercise_type)
                        
                        # Apply label augmentation
                        seg_labels = self.augmenter.augment(seg_labels)
                    
                    # Create windowed sequences
                    sequences, cls_labels, seg_window_labels = self._create_windowed_sequences(
                        features, seg_labels, exercise_idx
                    )
                    
                    if len(sequences) > 0:
                        exercise_sequences.extend(sequences)
                        exercise_cls_labels.extend(cls_labels)
                        exercise_seg_labels.extend(seg_window_labels)
                        print(f"    Created {len(sequences)} sequences from {video_file}")
                
                if len(exercise_sequences) > 0:
                    all_sequences.extend(exercise_sequences)
                    all_cls_labels.extend(exercise_cls_labels)
                    all_seg_labels.extend(exercise_seg_labels)
                    samples_per_exercise[exercise_type] = len(exercise_sequences)
                    print(f"  Total sequences for {exercise_type}: {len(exercise_sequences)}")
                else:
                    print(f"  No sequences created for {exercise_type}")
                    
            except Exception as e:
                print(f"  Error processing {exercise_type}: {e}")
                continue
        
        if len(all_sequences) == 0:
            raise ValueError("No valid sequences created from any exercise type")
        
        # Convert to numpy arrays
        X = np.array(all_sequences)  # Shape: (num_samples, window_size, num_features)
        y_classification = np.array(all_cls_labels)  # Shape: (num_samples,)
        y_segmentation = np.array(all_seg_labels)  # Shape: (num_samples, window_size)
        
        print(f"\nDataset built successfully:")
        print(f"  Total sequences: {len(X)}")
        print(f"  Sequence shape: {X.shape}")
        print(f"  Classification labels shape: {y_classification.shape}")
        print(f"  Segmentation labels shape: {y_segmentation.shape}")
        print(f"  Samples per exercise: {samples_per_exercise}")
        
        # Create metadata
        metadata = {
            'exercise_types': self.exercise_types,
            'num_classes': len(self.exercise_types),
            'samples_per_exercise': samples_per_exercise,
            'fps': self.fps,
            'window_size': self.window_size,
            'num_features': 25,  # 25 joint angles
            'processing_fps': self.fps,
            'input_dim': 25,  # 25 joint angles
        }
        
        return X, y_classification, y_segmentation, metadata
    
    def save(self, X: np.ndarray, y_classification: np.ndarray, y_segmentation: np.ndarray, 
             metadata: Dict, path: str = 'data/processed/multitask_dataset_30fps.npz'):
        """
        Save the multitask dataset to NPZ file.
        
        Args:
            X: Feature sequences
            y_classification: Classification labels
            y_segmentation: Segmentation labels
            metadata: Dataset metadata
            path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save dataset
        np.savez(path, 
                X=X, 
                y_classification=y_classification, 
                y_segmentation=y_segmentation,
                metadata=metadata)
        
        print(f"Multitask dataset saved to {path}")
        print(f"  Exercise types: {metadata['exercise_types']}")
        print(f"  Total samples: {len(X)}")
        print(f"  Processing FPS: {metadata['fps']}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build multitask training dataset for exercise detection'
    )
    parser.add_argument('--output', type=str, default='data/processed/multitask_dataset_30fps.npz',
                        help='Output dataset path')
    parser.add_argument('--videos_dir', type=str, default='data/raw',
                        help='Videos directory path')
    parser.add_argument('--labels_dir', type=str, default='data/labels',
                        help='Labels directory path')
    args = parser.parse_args()
    
    try:
        # Build multitask dataset
        builder = MultitaskDatasetBuilder(
            videos_dir=args.videos_dir,
            labels_dir=args.labels_dir
        )
        X, y_classification, y_segmentation, metadata = builder.build()
        builder.save(X, y_classification, y_segmentation, metadata, path=args.output)
        
        print(f"Successfully created multitask dataset: {args.output}")
        
    except Exception as e:
        print(f"Error building dataset: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
