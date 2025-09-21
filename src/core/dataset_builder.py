#!/usr/bin/env python3
"""
Data Processing Module

Handles feature extraction, dataset building, and data augmentation for both exercise repetition 
segmentation and exercise type classification.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, List
import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils.video_labeler import VideoSegmenter  # Removed - not needed for multitask


class LabelAugmenter:
    """
    Augments binary labels using Gaussian distribution around rep-start markers.
    
    This creates more realistic temporal patterns for better model training.
    """
    
    def __init__(self, fps: int = 30, margin_frames: int = 3, sigma: float = 2.9961):
        """
        Initialize the label augmenter.
        
        Args:
            fps: Frames per second of the videos
            margin_frames: Number of frames to expand around labels (±7)
            sigma: Standard deviation for Gaussian distribution
        """
        self.margin = margin_frames
        self.sigma = sigma

    def augment(self, labels: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian augmentation around positive samples.
        
        Args:
            labels: Binary array where 1 indicates repetition start
            
        Returns:
            Augmented labels with Gaussian-weighted positive regions
        """
        aug = labels.copy().astype(float)
        ones = np.where(labels == 1.0)[0]
        
        for idx in ones:
            start = max(0, idx - self.margin)
            end = min(len(labels), idx + self.margin + 1)
            
            # Create Gaussian weights
            x = np.arange(start, end) - idx
            weights = np.exp(-(x**2) / (2 * self.sigma**2))
            
            # Apply Gaussian weights (keep maximum values)
            aug[start:end] = np.maximum(aug[start:end], weights)
            
        return aug


class FeatureExtractor:
    """
    Extracts pose-based features from video frames using MediaPipe.
    
    Calculates 25 joint angles from pose landmarks to represent body posture.
    """
    
    def __init__(self):
        """Initialize MediaPipe pose detector and define joint angle calculations."""
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        
        # Relevant landmark indices for angle calculation
        self.relevant_ids = [
            2,   # left eye
            5,   # right eye
            11, 12, 13, 14, 15, 16,         # shoulders, elbows, wrists
            17, 18, 19, 20, 21, 22,         # fingers and thumbs
            23, 24, 25, 26, 27, 28,         # hips, knees, ankles
            29, 30, 31, 32                  # heels, foot indices
        ]

        # Angle triplets: (a, b, c) where angle is at point b between lines (a-b) and (b-c)
        self.angle_triplets = [
            # --- Head alignment (neck area approximation) ---
            (2, 0, 3),       # left shoulder - left eye - right shoulder
            (14, 2, 0),      # left hip - left shoulder - left eye
            (15, 3, 1),      # right hip - right shoulder - right eye

            # --- Shoulders ---
            (14, 2, 4),      # left hip - left shoulder - left elbow
            (15, 3, 5),      # right hip - right shoulder - right elbow

            # --- Elbows ---
            (2, 4, 6),       # left shoulder - left elbow - left wrist
            (3, 5, 7),       # right shoulder - right elbow - right wrist

            # --- Wrists (flexion/extension) ---
            (4, 6, 10),      # left elbow - wrist - index
            (5, 7, 11),      # right elbow - wrist - index
            (4, 6, 12),      # left elbow - wrist - thumb
            (5, 7, 13),      # right elbow - wrist - thumb
            (4, 6, 8),       # left elbow - wrist - pinky
            (5, 7, 9),       # right elbow - wrist - pinky

            # --- Spine & Hip connection ---
            (2, 14, 15),     # left shoulder - left hip - right hip
            (3, 15, 14),     # right shoulder - right hip - left hip

            # --- Hips ---
            (2, 14, 16),     # left shoulder - left hip - left knee
            (3, 15, 17),     # right shoulder - right hip - right knee

            # --- Knees ---
            (14, 16, 18),    # left hip - knee - ankle
            (15, 17, 19),    # right hip - knee - ankle

            # --- Ankles / Feet ---
            (16, 18, 22),    # left knee - ankle - foot index
            (17, 19, 23),    # right knee - ankle - foot index
            (18, 20, 22),    # left ankle - heel - foot index
            (19, 21, 23),    # right ankle - heel - foot index
            (20, 18, 16),    # heel - ankle - knee
            (21, 19, 17),    # heel - ankle - knee
        ]

    def extract_angles(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Extract joint angles from a video frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Array of 25 normalized joint angles or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        
        if not res.pose_landmarks:
            return None
            
        # Extract relevant landmarks
        lm_map = [res.pose_landmarks.landmark[idx] for idx in self.relevant_ids]
        
        # Calculate angles for each triplet
        angles = []
        for a_id, b_id, c_id in self.angle_triplets:
            a = lm_map[a_id]
            b = lm_map[b_id]
            c = lm_map[c_id]
            
            # Calculate distances
            ab = np.hypot(a.x - b.x, a.y - b.y)
            bc = np.hypot(b.x - c.x, b.y - c.y)
            ac = np.hypot(a.x - c.x, a.y - c.y)
            
            # Calculate angle using cosine law
            if ab == 0 or bc == 0:
                angle = 0.0
            else:
                cos_val = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
                angle = float(np.arccos(np.clip(cos_val, -1.0, 1.0))) / np.pi  # Normalize to [0, 1]
                
            angles.append(angle)
            
        return np.array(angles) if angles else None

    def get_feature_dimension(self) -> int:
        """Get the number of features extracted per frame."""
        return len(self.angle_triplets)


class SegmentationDatasetBuilder:
    """
    Builds training datasets for exercise repetition segmentation.
    
    Processes videos, extracts features, and creates labeled datasets for training
    exercise-specific repetition detection models.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", fps: int = 30):
        """
        Initialize the segmentation dataset builder.
        
        Args:
            videos_dir: Directory containing training videos
            labels_dir: Directory containing manual labels
            fps: Frames per second for temporal calculations
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.augmenter = LabelAugmenter(fps, margin_frames=7, sigma=2.0)  # Increased from 0.1s to 0.3s
        self.extractor = FeatureExtractor()
        # # self.segmenter = VideoSegmenter(videos_dir=videos_dir, labels_dir=labels_dir)  # Not needed for multitask  # Not needed for multitask

    def build(self, exercise_type: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Build segmentation dataset from labeled videos for a specific exercise.
        
        Args:
            exercise_type: Specific exercise type to process
            
        Returns:
            Tuple of (features, labels) arrays
        """
        X_all, y_all = [], []
        
        # Get video directory for this exercise
        video_dir = os.path.join(self.videos_dir, exercise_type)
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"Processing {len(video_files)} videos for {exercise_type}...")
        
        for video_file in video_files:
            try:
                # Load labeled frames
                labels, frames = zip(*self.segmenter.get_labeled_frames(f"{exercise_type}/{video_file}"))
                labels = self.augmenter.augment(np.array(labels))

                # Extract features for each frame
                for idx, frame in enumerate(frames):
                    feat = self.extractor.extract_angles(frame)
                    if feat is not None:
                        X_all.append(feat)
                        y_all.append(labels[idx])

                print(f"Processed {video_file}: {len(frames)} frames")
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

        if not X_all:
            raise ValueError("No valid features extracted from videos")

        X = np.stack(X_all)  # shape: (total_frames, num_features)
        y = np.array(y_all)  # shape: (total_frames,)
        
        print(f"Built segmentation dataset: {X.shape[0]} frames with {X.shape[1]} features each")
        print(f"Positive samples: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
        
        return X, y

    def save(self, X: np.ndarray, y: np.ndarray, path: str = 'dataset.npz'):
        """
        Save dataset to NPZ file.
        
        Args:
            X: Feature array
            y: Label array
            path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        np.savez(path, X=X, y=y)
        print(f"Segmentation dataset saved to {path}")


class ClassificationDatasetBuilder:
    """
    Builds training datasets for exercise type classification.
    Processes all videos from all exercise types to create a unified classification dataset.
    """
    
    def __init__(self, videos_dir: str = "data/raw", fps: int = 30, window_size: int = 30):
        """
        Initialize the classification dataset builder.
        Args:
            videos_dir: Directory containing training videos organized by exercise type
            fps: Frames per second for temporal calculations
            window_size: Number of frames per classification sample (default: 30)
        """
        self.videos_dir = videos_dir
        self.extractor = FeatureExtractor()
        self.window_size = window_size

    def build(self) -> tuple[np.ndarray, np.ndarray, dict, list]:
        """
        Build classification dataset from all exercise videos using sliding windows.
        Returns:
            Tuple of (features, labels, label_mapping, class_names) where:
            - features: Array of shape (num_samples, window_size, num_features)
            - labels: Array of shape (num_samples,) with integer labels
            - label_mapping: Dict mapping exercise names to integer labels
            - class_names: List of class names ordered by label index
        """
        X_all, y_all = [], []
        label_mapping = {}
        current_label = 0
        
        # Get all exercise types (subdirectories)
        exercise_types = [d for d in os.listdir(self.videos_dir) 
                         if os.path.isdir(os.path.join(self.videos_dir, d))]
        exercise_types.sort()  # Ensure consistent ordering
        
        print(f"Found {len(exercise_types)} exercise types: {exercise_types}")
        
        for exercise_type in exercise_types:
            label_mapping[exercise_type] = current_label
            video_dir = os.path.join(self.videos_dir, exercise_type)
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            
            print(f"Processing {len(video_files)} videos for {exercise_type} (label: {current_label})...")
            
            for video_file in video_files:
                try:
                    video_path = os.path.join(video_dir, video_file)
                    cap = cv2.VideoCapture(video_path)
                    features = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        feat = self.extractor.extract_angles(frame)
                        if feat is not None:
                            features.append(feat)
                    cap.release()
                    if len(features) < self.window_size:
                        print(f"  Skipping {video_file}: not enough frames ({len(features)})")
                        continue
                    # Create sliding windows
                    for i in range(len(features) - self.window_size + 1):
                        window = features[i:i+self.window_size]
                        X_all.append(window)
                        y_all.append(current_label)
                    print(f"  Processed {video_file}: {len(features)} frames, {len(features) - self.window_size + 1} windows")
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    continue
            current_label += 1

        if not X_all:
            raise ValueError("No valid features extracted from videos")

        X = np.stack(X_all)  # shape: (num_samples, window_size, num_features)
        y = np.array(y_all)  # shape: (num_samples,)
        
        # Build class_names list ordered by label index
        class_names = [None] * len(label_mapping)
        for name, idx in label_mapping.items():
            class_names[idx] = name
        
        print(f"Built classification dataset: {X.shape[0]} windows of {X.shape[1]} frames each, {X.shape[2]} features per frame")
        print(f"Exercise distribution:")
        for exercise, label in label_mapping.items():
            count = np.sum(y == label)
            print(f"  {exercise}: {count} windows ({count/len(y)*100:.1f}%)")
        
        return X, y, label_mapping, class_names

    def save(self, X: np.ndarray, y: np.ndarray, label_mapping: dict, class_names: list, path: str = 'classification_dataset.npz'):
        """
        Save classification dataset to NPZ file.
        Args:
            X: Feature array
            y: Label array
            label_mapping: Mapping of exercise names to integer labels
            class_names: List of class names ordered by label index
            path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        # Save both data and label mapping (note: load with allow_pickle=True)
        np.savez(path, X=X, y=y, label_mapping=label_mapping, class_names=class_names)
        print(f"Classification dataset saved to {path}")
        print(f"Label mapping: {label_mapping}")
        print(f"Class names: {class_names}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build training datasets for exercise detection'
    )
    parser.add_argument('--mode', type=str, required=True, choices=['segmentation', 'classification'],
                        help='Dataset mode: segmentation or classification')
    parser.add_argument('--exercise', type=str, default=None,
                        help='Exercise type for segmentation mode (e.g., push-ups, squats)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output dataset path')
    args = parser.parse_args()
    
    try:
        if args.mode == 'segmentation':
            if not args.exercise:
                raise ValueError("Exercise type required for segmentation mode")
            
            # Build segmentation dataset
            if args.output is None:
                args.output = f'data/processed/dataset_{args.exercise}.npz'
            
            builder = SegmentationDatasetBuilder()
            X, y = builder.build(exercise_type=args.exercise)
            builder.save(X, y, path=args.output)
            
        elif args.mode == 'classification':
            # Build classification dataset
            if args.output is None:
                args.output = 'data/processed/classification_dataset.npz'
            
            builder = ClassificationDatasetBuilder()
            X, y, label_mapping, class_names = builder.build()
            builder.save(X, y, label_mapping, class_names, path=args.output)
        
        print(f"Successfully created {args.mode} dataset: {args.output}")
        
    except Exception as e:
        print(f"Error building dataset: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

class MultitaskDatasetBuilder:
    """
    Multitask dataset builder that handles both exercise and no-exercise videos.
    """
    
    def __init__(self, videos_dir: str = "data/raw", no_exercise_dir: str = "data/no_exercise", 
                 labels_dir: str = "data/labels", fps: int = 30, window_size: int = 30, 
                 margin_frames: int = 12, sigma: float = 3.40, no_exercise_ratio: float = 0.3):
        """
        Initialize the multitask dataset builder.
        
        Args:
            videos_dir: Directory containing exercise videos
            no_exercise_dir: Directory containing no-exercise videos
            labels_dir: Directory containing manual labels (CSV files)
            fps: Video frame rate (default: 30)
            window_size: Number of frames per sequence (default: 30)
            margin_frames: Number of frames to expand around labels (±12)
            sigma: Standard deviation for Gaussian distribution
            no_exercise_ratio: Ratio of no-exercise samples to include (0.0-1.0)
        """
        self.videos_dir = videos_dir
        self.no_exercise_dir = no_exercise_dir
        self.labels_dir = labels_dir
        self.fps = fps
        self.window_size = window_size
        self.no_exercise_ratio = no_exercise_ratio
        
        # Initialize feature extractor and label augmenter
        self.extractor = FeatureExtractor()
        self.augmenter = LabelAugmenter(fps=fps, margin_frames=margin_frames, sigma=sigma)
        
        # Exercise types including no-exercise
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips', 'no_exercise']
        self.no_exercise_idx = len(self.exercise_types) - 1  # Last index is no-exercise
        
        print(f"Multitask dataset builder initialized:")
        print(f"  Window size: {window_size} frames ({window_size/fps:.1f} seconds)")
        print(f"  FPS: {fps}")
        print(f"  No-exercise ratio: {no_exercise_ratio}")
        print(f"  Exercise types: {self.exercise_types}")
    
    def build_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build the multitask dataset with no-exercise support.
        
        Returns:
            Tuple of (X, y_classification, y_segmentation, metadata)
        """
        print("Building multitask dataset...")
        
        all_sequences = []
        all_cls_labels = []
        all_seg_labels = []
        video_metadata = []
        
        # Process exercise videos
        print("\nProcessing exercise videos...")
        exercise_videos = self._get_video_files(self.videos_dir)
        
        for video_path in exercise_videos:
            print(f"Processing exercise video: {os.path.basename(video_path)}")
            
            # Determine exercise type from directory structure
            exercise_type = self._get_exercise_type_from_path(video_path)
            if exercise_type not in self.exercise_types[:-1]:  # Exclude no_exercise
                print(f"  Skipping unknown exercise type: {exercise_type}")
                continue
            
            exercise_idx = self.exercise_types.index(exercise_type)
            
            # Extract features
            features = self._extract_video_features(video_path)
            if not features:
                print(f"  No features extracted, skipping")
                continue
            
            # Load labels
            label_path = os.path.join(self.labels_dir, exercise_type, f"{os.path.splitext(os.path.basename(video_path))[0]}.csv")
            labels = self._load_labels_csv(label_path, max(features.keys()), is_no_exercise=False)
            
            # Skip videos with no positive labels (except no-exercise videos)
            if not any(labels == 1.0) and exercise_type != 'no_exercise':
                print(f'  No positive labels found, skipping video')
                continue
            
            # Create sequences
            sequences, cls_labels, seg_labels = self._create_windowed_sequences(
                features, labels, exercise_idx, is_no_exercise=False
            )
            
            all_sequences.extend(sequences)
            all_cls_labels.extend(cls_labels)
            all_seg_labels.extend(seg_labels)
            
            video_metadata.append({
                'video_path': video_path,
                'exercise_type': exercise_type,
                'sequences_count': len(sequences),
                'is_no_exercise': False
            })
            
            print(f"  Created {len(sequences)} sequences")
        
        # Process no-exercise videos
        print("\nProcessing no-exercise videos...")
        no_exercise_videos = self._get_video_files(self.no_exercise_dir)
        
        # Limit no-exercise videos based on ratio
        max_no_exercise = int(len(all_sequences) * self.no_exercise_ratio / (1 - self.no_exercise_ratio))
        no_exercise_videos = no_exercise_videos[:max_no_exercise]
        
        for video_path in no_exercise_videos:
            print(f"Processing no-exercise video: {os.path.basename(video_path)}")
            
            # Extract features
            features = self._extract_video_features(video_path)
            if not features:
                print(f"  No features extracted, skipping")
                continue
            
            # Create all-zero labels for no-exercise videos
            labels = np.zeros(max(features.keys()) + 1)
            
            # Create sequences
            sequences, cls_labels, seg_labels = self._create_windowed_sequences(
                features, labels, self.no_exercise_idx, is_no_exercise=True
            )
            
            all_sequences.extend(sequences)
            all_cls_labels.extend(cls_labels)
            all_seg_labels.extend(seg_labels)
            
            video_metadata.append({
                'video_path': video_path,
                'exercise_type': 'no_exercise',
                'sequences_count': len(sequences),
                'is_no_exercise': True
            })
            
            print(f"  Created {len(sequences)} sequences")
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y_classification = np.array(all_cls_labels)
        y_segmentation = np.array(all_seg_labels)
        
        # Create metadata
        metadata = {
            'total_sequences': len(all_sequences),
            'exercise_types': self.exercise_types,
            'window_size': self.window_size,
            'fps': self.fps,
            'no_exercise_ratio': self.no_exercise_ratio,
            'video_metadata': video_metadata,
            'class_distribution': {
                exercise_type: np.sum(y_classification == i) 
                for i, exercise_type in enumerate(self.exercise_types)
            }
        }
        
        print(f"\nDataset built successfully!")
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Class distribution: {metadata['class_distribution']}")
        
        return X, y_classification, y_segmentation, metadata
    
    def _extract_video_features(self, video_path: str) -> Dict[int, np.ndarray]:
        """Extract features from ALL video frames for windowing."""
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
                import pandas as pd
                df = pd.read_csv(label_path)
                labels = np.zeros(max_frame_idx + 1)
                
                if 'label' in df.columns and 'frame' in df.columns:
                    frame_indices = df['frame'].values
                    rep_labels = df['label'].values
                    
                    positive_count = 0
                    for idx, label_val in zip(frame_indices, rep_labels):
                        if label_val == 1.0 and 0 <= idx <= max_frame_idx:
                            labels[idx] = 1.0
                            positive_count += 1
                    
                    if positive_count > 0:
                        print(f"    Found {positive_count} rep markers in CSV")
                        return labels
                        
            except Exception as e:
                print(f"  Error loading labels from {label_path}: {e}")
        
        print(f"  No valid labels found, using all zeros")
        return np.zeros(max_frame_idx + 1)
    
    def _create_windowed_sequences(self, features: Dict[int, np.ndarray], labels: np.ndarray, 
                                 exercise_idx: int, is_no_exercise: bool = False) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
        """Create windowed sequences using sliding window."""
        max_frame = max(features.keys()) if features else 0
        sequences = []
        cls_labels = []
        seg_labels = []
        
        for start_frame in range(max_frame - self.window_size + 2):
            frame_indices = list(range(start_frame, start_frame + self.window_size))
            
            if all(idx in features and idx < len(labels) for idx in frame_indices):
                window_features = []
                window_labels = []
                
                for frame_idx in frame_indices:
                    window_features.append(features[frame_idx])
                    window_labels.append(labels[frame_idx])
                
                sequences.append(np.array(window_features))
                cls_labels.append(exercise_idx)
                
                if not is_no_exercise:
                    augmented_labels = self.augmenter.augment(np.array(window_labels))
                else:
                    augmented_labels = np.array(window_labels)
                
                seg_labels.append(augmented_labels)
        
        return sequences, cls_labels, seg_labels
    
    def _get_video_files(self, directory: str) -> List[str]:
        """Get all video files from directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return video_files
    
    def _get_exercise_type_from_path(self, video_path: str) -> str:
        """Extract exercise type from video path."""
        path_parts = video_path.split(os.sep)
        for part in path_parts:
            if part in self.exercise_types[:-1]:
                return part
        
        return self.exercise_types[0]
