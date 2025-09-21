#!/usr/bin/env python3
"""
Improved Two-Stage Dataset Builder

Stage 1: Process videos to extract angles and labels with augmentation (parameter-independent)
Stage 2: Create windows from processed data with configurable size and stride

This allows experimenting with different window parameters without re-processing videos.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import cv2
import mediapipe as mp
import pickle
import pandas as pd
from pathlib import Path
import json

# Import existing classes from the original dataset builder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.dataset_builder import FeatureExtractor, LabelAugmenter


class VideoProcessor:
    """
    Stage 1: Process videos to extract pose angles and labels with augmentation.
    This stage is independent of window parameters.
    """
    
    def __init__(self, videos_dir: str = "data/raw", no_exercise_dir: str = "data/no_exercise", 
                 labels_dir: str = "data/labels", fps: int = 30, margin_frames: int = 7, sigma: float = 3.0):
        """
        Initialize the video processor.
        
        Args:
            videos_dir: Directory containing exercise videos
            no_exercise_dir: Directory containing no-exercise videos  
            labels_dir: Directory containing manual labels (CSV files)
            fps: Video frame rate
            margin_frames: Number of frames to expand around labels (±7)
            sigma: Standard deviation for Gaussian distribution
        """
        self.videos_dir = videos_dir
        self.no_exercise_dir = no_exercise_dir
        self.labels_dir = labels_dir
        self.fps = fps
        
        # Initialize components
        self.extractor = FeatureExtractor()
        self.augmenter = LabelAugmenter(fps=fps, margin_frames=7, sigma=2.0)
        
        # Exercise types
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips', 'no_exercise']
        self.no_exercise_idx = len(self.exercise_types) - 1
        
        print(f"Video processor initialized:")
        print(f"  FPS: {fps}")
        print(f"  Label augmentation: {margin_frames} frames (σ={sigma})")
        print(f"  Exercise types: {self.exercise_types}")
    
    def process_all_videos(self, output_dir: str = "data/processed") -> str:
        """
        Process all videos and save the extracted features and labels.
        
        Args:
            output_dir: Directory to save processed data
            
        Returns:
            Path to the saved processed data file
        """
        print("="*60)
        print("STAGE 1: PROCESSING VIDEOS TO EXTRACT FEATURES AND LABELS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_video_data = []
        
        # Process exercise videos
        print("\nProcessing exercise videos...")
        exercise_videos = self._get_video_files(self.videos_dir)
        
        for video_path in exercise_videos:
            print(f"\nProcessing: {os.path.basename(video_path)}")
            
            # Determine exercise type
            exercise_type = self._get_exercise_type_from_path(video_path)
            if exercise_type not in self.exercise_types[:-1]:  # Exclude no_exercise
                print(f"  Skipping unknown exercise type: {exercise_type}")
                continue
                
            exercise_idx = self.exercise_types.index(exercise_type)
            
            # Extract features from all frames
            print("  Extracting features...")
            features = self._extract_video_features(video_path)
            if not features:
                print("    No features extracted, skipping")
                continue
                
            print(f"    Extracted features for {len(features)} frames")
            
            # Load and augment labels
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label_path = os.path.join(self.labels_dir, exercise_type, f'{video_name}.csv')
            labels = self._load_labels_csv(label_path, max(features.keys()), is_no_exercise=False)
            
            # Skip videos with no positive labels
            if not any(labels == 1.0):
                print(f'    No positive labels found, skipping video')
                continue
                
            print(f"    Loaded {len(labels)} labels, {sum(labels == 1.0)} positive")
            
            # Store video data
            video_data = {
                'video_path': video_path,
                'exercise_type': exercise_type,
                'exercise_idx': exercise_idx,
                'features': features,  # Dict mapping frame_idx -> feature_vector
                'labels': labels,      # Array of labels for each frame
                'is_no_exercise': False,
                'fps': self.fps
            }
            
            all_video_data.append(video_data)
            print(f"    Video processed successfully")
        
        # Process no-exercise videos
        print("\nProcessing no-exercise videos...")
        no_exercise_videos = self._get_video_files(self.no_exercise_dir)
        
        for video_path in no_exercise_videos:
            print(f"\nProcessing no-exercise: {os.path.basename(video_path)}")
            
            # Extract features
            features = self._extract_video_features(video_path)
            if not features:
                print("    No features extracted, skipping")
                continue
                
            print(f"    Extracted features for {len(features)} frames")
            
            # Create all-zero labels
            labels = np.zeros(max(features.keys()) + 1)
            
            # Store video data
            video_data = {
                'video_path': video_path,
                'exercise_type': 'no_exercise',
                'exercise_idx': self.no_exercise_idx,
                'features': features,
                'labels': labels,
                'is_no_exercise': True,
                'fps': self.fps
            }
            
            all_video_data.append(video_data)
            print(f"    No-exercise video processed successfully")
        
        # Save processed data
        output_file = os.path.join(output_dir, "processed_videos.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(all_video_data, f)
            
        # Also save metadata as JSON for inspection
        metadata = {
            'total_videos': len(all_video_data),
            'exercise_types': self.exercise_types,
            'fps': self.fps,
            'videos': [
                {
                    'video_path': vd['video_path'],
                    'exercise_type': vd['exercise_type'],
                    'total_frames': len(vd['features']),
                    'positive_labels': int(sum(vd['labels'] == 1.0)) if not vd['is_no_exercise'] else 0
                }
                for vd in all_video_data
            ]
        }
        
        metadata_file = os.path.join(output_dir, "processed_videos_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n" + "="*60)
        print("STAGE 1 COMPLETED!")
        print("="*60)
        print(f"Processed {len(all_video_data)} videos")
        print(f"Saved to: {output_file}")
        print(f"Metadata: {metadata_file}")
        
        return output_file
    
    def _get_video_files(self, directory: str) -> List[str]:
        """Get all video files from directory."""
        video_files = []
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        video_files.append(os.path.join(root, file))
        return video_files
    
    def _get_exercise_type_from_path(self, video_path: str) -> str:
        """Extract exercise type from video path."""
        path_parts = os.path.normpath(video_path).split(os.sep)
        for part in path_parts:
            if part in self.exercise_types[:-1]:  # Exclude no_exercise
                return part
        return "unknown"
    
    def _extract_video_features(self, video_path: str) -> Dict[int, np.ndarray]:
        """Extract features from all video frames."""
        cap = cv2.VideoCapture(video_path)
        features = {}
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                feat = self.extractor.extract_angles(frame)
                if feat is not None:
                    features[frame_count] = feat
                
                frame_count += 1
        finally:
            cap.release()
        
        return features
    
    def _load_labels_csv(self, label_path: str, max_frame_idx: int, is_no_exercise: bool = False) -> np.ndarray:
        """Load segmentation labels from CSV file."""
        if is_no_exercise:
            return np.zeros(max_frame_idx + 1)
        
        if os.path.exists(label_path):
            try:
                df = pd.read_csv(label_path)
                labels = np.zeros(max_frame_idx + 1)
                
                if 'label' in df.columns and 'frame' in df.columns:
                    frame_indices = df['frame'].values
                    label_values = df['label'].values
                    
                    # Only use frames that exist in the video
                    valid_indices = frame_indices[frame_indices <= max_frame_idx]
                    valid_labels = label_values[frame_indices <= max_frame_idx]
                    
                    labels[valid_indices] = valid_labels
                    
                    # Apply augmentation
                    labels = self.augmenter.augment(labels)
                
                return labels
            except Exception as e:
                print(f"Error loading labels from {label_path}: {e}")
                return np.zeros(max_frame_idx + 1)
        else:
            print(f"Label file not found: {label_path}")
            return np.zeros(max_frame_idx + 1)


class ConfigurableDatasetBuilder:
    """
    Stage 2: Create windowed datasets from processed video data.
    Configurable window size and stride.
    """
    
    def __init__(self, window_size: int = 30, stride: int = 2, no_exercise_ratio: float = 0.3):
        """
        Initialize the dataset builder.
        
        Args:
            window_size: Number of frames per sequence
            stride: Step size for sliding windows
            no_exercise_ratio: Ratio of no-exercise samples to include
        """
        self.window_size = window_size
        self.stride = stride
        self.no_exercise_ratio = no_exercise_ratio
        
        print(f"Configurable dataset builder initialized:")
        print(f"  Window size: {window_size} frames")
        print(f"  Stride: {stride} frames")
        print(f"  No-exercise ratio: {no_exercise_ratio}")
    
    def build_dataset(self, processed_data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build dataset from processed video data.
        
        Args:
            processed_data_file: Path to processed video data
            
        Returns:
            Tuple of (X, y_classification, y_segmentation, metadata)
        """
        print("="*60)
        print(f"STAGE 2: BUILDING DATASET (window_size={self.window_size}, stride={self.stride})")
        print("="*60)
        
        # Load processed data
        print(f"Loading processed data from: {processed_data_file}")
        with open(processed_data_file, 'rb') as f:
            all_video_data = pickle.load(f)
        
        print(f"Loaded {len(all_video_data)} processed videos")
        
        all_sequences = []
        all_cls_labels = []
        all_seg_labels = []
        video_metadata = []
        
        # Process exercise videos
        exercise_videos = [vd for vd in all_video_data if not vd['is_no_exercise']]
        no_exercise_videos = [vd for vd in all_video_data if vd['is_no_exercise']]
        
        print(f"\nProcessing {len(exercise_videos)} exercise videos...")
        
        for video_data in exercise_videos:
            print(f"Creating windows for: {os.path.basename(video_data['video_path'])}")
            
            sequences, cls_labels, seg_labels = self._create_windowed_sequences(
                video_data['features'], 
                video_data['labels'], 
                video_data['exercise_idx']
            )
            
            if sequences:
                all_sequences.extend(sequences)
                all_cls_labels.extend(cls_labels)
                all_seg_labels.extend(seg_labels)
                
                video_metadata.append({
                    'video_path': video_data['video_path'],
                    'exercise_type': video_data['exercise_type'],
                    'sequences_count': len(sequences),
                    'is_no_exercise': False
                })
                
                print(f"  Created {len(sequences)} sequences")
            else:
                print(f"  No sequences created (video too short)")
        
        # Determine how many no-exercise videos to include
        total_exercise_sequences = len(all_sequences)
        max_no_exercise_sequences = int(total_exercise_sequences * self.no_exercise_ratio / (1 - self.no_exercise_ratio))
        
        print(f"\nProcessing no-exercise videos (max {max_no_exercise_sequences} sequences)...")
        
        no_exercise_sequences_added = 0
        for video_data in no_exercise_videos:
            if no_exercise_sequences_added >= max_no_exercise_sequences:
                break
                
            print(f"Creating windows for: {os.path.basename(video_data['video_path'])}")
            
            sequences, cls_labels, seg_labels = self._create_windowed_sequences(
                video_data['features'], 
                video_data['labels'], 
                video_data['exercise_idx']
            )
            
            # Limit sequences to not exceed target ratio
            sequences_to_add = min(len(sequences), max_no_exercise_sequences - no_exercise_sequences_added)
            if sequences_to_add > 0:
                all_sequences.extend(sequences[:sequences_to_add])
                all_cls_labels.extend(cls_labels[:sequences_to_add])
                all_seg_labels.extend(seg_labels[:sequences_to_add])
                
                video_metadata.append({
                    'video_path': video_data['video_path'],
                    'exercise_type': video_data['exercise_type'],
                    'sequences_count': sequences_to_add,
                    'is_no_exercise': True
                })
                
                no_exercise_sequences_added += sequences_to_add
                print(f"  Added {sequences_to_add} sequences")
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y_classification = np.array(all_cls_labels)
        y_segmentation = np.array(all_seg_labels)
        
        # Create metadata
        exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips', 'no_exercise']
        metadata = {
            'total_sequences': len(all_sequences),
            'exercise_types': exercise_types,
            'window_size': self.window_size,
            'stride': self.stride,
            'no_exercise_ratio': self.no_exercise_ratio,
            'video_metadata': video_metadata,
            'class_distribution': {
                exercise_type: np.sum(y_classification == i) 
                for i, exercise_type in enumerate(exercise_types)
            }
        }
        
        print(f"\n" + "="*60)
        print("STAGE 2 COMPLETED!")
        print("="*60)
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Class distribution: {metadata['class_distribution']}")
        
        return X, y_classification, y_segmentation, metadata
    
    def _create_windowed_sequences(self, features: Dict[int, np.ndarray], labels: np.ndarray, 
                                 exercise_idx: int) -> Tuple[List, List, List]:
        """Create sliding window sequences from features and labels."""
        sequences = []
        cls_labels = []
        seg_labels = []
        
        # Get sorted frame indices
        frame_indices = sorted(features.keys())
        max_frame = max(frame_indices) if frame_indices else 0
        
        # Create sliding windows
        for start_idx in range(0, max_frame - self.window_size + 2, self.stride):
            end_idx = start_idx + self.window_size
            
            # Check if we have features for all frames in the window
            window_frames = list(range(start_idx, end_idx))
            if all(frame in features for frame in window_frames):
                # Extract features for this window
                window_features = [features[frame] for frame in window_frames]
                window_sequence = np.array(window_features)
                
                # Extract labels for this window
                window_labels = labels[start_idx:end_idx] if end_idx <= len(labels) else labels[start_idx:]
                if len(window_labels) < self.window_size:
                    # Pad with zeros if needed
                    padded_labels = np.zeros(self.window_size)
                    padded_labels[:len(window_labels)] = window_labels
                    window_labels = padded_labels
                
                sequences.append(window_sequence)
                cls_labels.append(exercise_idx)
                seg_labels.append(window_labels[:self.window_size])
        
        return sequences, cls_labels, seg_labels
    
    def save_dataset(self, X: np.ndarray, y_classification: np.ndarray, y_segmentation: np.ndarray, 
                    metadata: Dict[str, Any], output_dir: str) -> str:
        """Save the dataset to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create descriptive filename
        filename = f"dataset_w{self.window_size}_s{self.stride}.npz"
        output_file = os.path.join(output_dir, filename)
        
        # Save dataset
        np.savez_compressed(output_file, 
                           X=X, 
                           y_classification=y_classification, 
                           y_segmentation=y_segmentation)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"metadata_w{self.window_size}_s{self.stride}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Dataset saved to: {output_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return output_file


if __name__ == "__main__":
    # Example usage
    print("Two-Stage Dataset Builder")
    print("This should be used via the experiment script")
