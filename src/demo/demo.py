#!/usr/bin/env python3
"""
Configurable Video Demo for Workout Tracker

This demo processes a video using the trained multitask TCN model with configurable
window size and stride parameters, and generates:
1. Comprehensive plots showing model outputs over time
2. Output video with frame-by-frame predictions and exercise counters
3. Configurable window processing for efficiency

The demo supports configurable window size and stride for different model configurations.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.dataset_builder import FeatureExtractor
from training.model import build_multitask_tcn_model


class SignalProcessor:
    """Handles signal processing and peak detection for rep counting."""
    
    def __init__(self, min_peak_threshold: float = 0.3, smoothing_sigma: float = 0.8):
        """
        Initialize signal processor.
        
        Args:
            min_peak_threshold: Minimum threshold for peak detection
            smoothing_sigma: Sigma for Gaussian smoothing
        """
        self.min_peak_threshold = min_peak_threshold
        self.smoothing_sigma = smoothing_sigma
    
    def smooth_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to the signal."""
        return gaussian_filter1d(signal_data, sigma=self.smoothing_sigma)


class VideoProcessor:
    """Processes videos using the trained multitask TCN model with configurable parameters."""
    
    def __init__(self, model_path: str, window_size: int = 30, stride: int = 1):
        """
        Initialize video processor.
        
        Args:
            model_path: Path to the trained model
            window_size: Size of sliding window (30 frames = 1 second at 30fps)
            stride: Stride for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_extractor = FeatureExtractor()
        self.signal_processor = SignalProcessor()
        
        # Exercise types (must match training)
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips']
        
        # Load model
        print(f"Loading model from {model_path}...")
        print(f"Using window_size: {window_size}, stride: {stride}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    
    def extract_features_from_video(self, video_path: str) -> Tuple[Dict[int, np.ndarray], int]:
        """
        Extract features from all frames in the video.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Tuple of (features_dict, total_frames)
        """
        cap = cv2.VideoCapture(video_path)
        features = {}
        frame_count = 0
        
        print(f"Extracting features from {video_path}...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract features for this frame
                frame_features = self.feature_extractor.extract_angles(frame)
                if frame_features is not None:
                    features[frame_count] = frame_features
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
        
        finally:
            cap.release()
        
        print(f"Feature extraction complete: {len(features)} valid frames out of {frame_count}")
        return features, frame_count
    
    def create_sliding_windows(self, features: Dict[int, np.ndarray], windows_per_second: int = 0) -> List[Tuple[int, np.ndarray]]:
        """
        Create sliding windows from features with configurable stride.
        
        Args:
            features: Dictionary mapping frame_index -> feature_array
            windows_per_second: Number of windows to process per second (0 = all windows)
            
        Returns:
            List of (start_frame, window_features) tuples
        """
        max_frame = max(features.keys()) if features else 0
        windows = []
        
        if windows_per_second == 0:
            # Use the configured stride
            step_size = self.stride
        else:
            # Process specific number of windows per second
            step_size = 30 // windows_per_second if windows_per_second > 0 else self.stride
        
        for start_frame in range(0, max_frame - self.window_size + 1, step_size):
            frame_indices = list(range(start_frame, start_frame + self.window_size))
            
            # Check if all frames in this window are available
            if all(idx in features for idx in frame_indices):
                window_features = np.array([features[idx] for idx in frame_indices])
                windows.append((start_frame, window_features))
        
        print(f"Created {len(windows)} sliding windows (window_size: {self.window_size}, stride: {self.stride}, step_size: {step_size})")
        return windows
    
    def predict_windows(self, windows: List[Tuple[int, np.ndarray]]) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
        """
        Run model predictions on all windows.
        
        Args:
            windows: List of (start_frame, window_features) tuples
            
        Returns:
            Tuple of (start_frames, classifications, segmentations)
        """
        if not windows:
            return [], [], []
        
        print(f"Running predictions on {len(windows)} windows...")
        
        # Prepare batch data
        window_data = np.array([window[1] for window in windows])
        start_frames = [window[0] for window in windows]
        
        # Run predictions
        predictions = self.model.predict(window_data, verbose=1)
        
        # Handle both single-task and multitask models
        if isinstance(predictions, list) and len(predictions) == 2:
            classifications, segmentations = predictions
        else:
            # Single output - assume it's segmentation
            classifications = np.zeros((len(windows), len(self.exercise_types)))
            segmentations = predictions
        
        print(f"Predictions complete: {len(start_frames)} windows processed")
        return start_frames, classifications, segmentations


class PlotGenerator:
    """Generates comprehensive plots for model outputs."""
    
    def __init__(self, exercise_types: List[str]):
        """Initialize plot generator."""
        self.exercise_types = exercise_types
    
    def generate_analysis_plot(self, 
                             start_frames: List[int], 
                             classifications: List[np.ndarray], 
                             segmentations: List[np.ndarray],
                             full_signal: np.ndarray,
                             peaks: np.ndarray,
                             rep_count: int,
                             output_path: str) -> None:
        """Generate comprehensive analysis plot."""
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot 1: Classification probabilities over time
        if len(classifications) > 0:
            classification_data = np.array(classifications)
            for i, exercise in enumerate(self.exercise_types):
                axes[0].plot(start_frames, classification_data[:, i], 
                            label=exercise, linewidth=2)
        
        axes[0].set_title('Exercise Classification Probabilities Over Time')
        axes[0].set_xlabel('Frame Number')
        axes[0].set_ylabel('Probability')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Segmentation signal and peaks
        axes[1].plot(full_signal, label='Rep Probability Signal', color='blue', alpha=0.7)
        axes[1].plot(peaks, full_signal[peaks], 'ro', markersize=8, label=f'Detected Peaks ({rep_count})')
        axes[1].set_title('Rep Detection Signal and Peak Detection')
        axes[1].set_xlabel('Frame Number')
        axes[1].set_ylabel('Rep Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to: {output_path}")


class VideoAnnotator:
    """Handles video annotation with predictions and rep counting."""
    
    def __init__(self, exercise_types: List[str]):
        """Initialize video annotator."""
        self.exercise_types = exercise_types
    
    def annotate_video(self, 
                      input_path: str, 
                      output_path: str,
                      start_frames: List[int],
                      classifications: List[np.ndarray],
                      segmentations: List[np.ndarray],
                      full_signal: np.ndarray,
                      peaks: np.ndarray,
                      fps: int = 30) -> None:
        """Annotate video with predictions and rep counting."""
        
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_idx = 0
        rep_count = 0
        
        print(f"Annotating video: {input_path} -> {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get predictions for this frame
                exercise, exercise_prob, rep_prob = self._get_frame_predictions(
                    frame_idx, start_frames, classifications, segmentations, full_signal
                )
                
                # Update rep count if we're at a peak
                if frame_idx in peaks:
                    rep_count += 1
                
                # Annotate frame
                self._annotate_frame(frame, exercise, exercise_prob, rep_prob, 
                                   rep_count, frame_idx, peaks)
                
                out.write(frame)
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"Annotated {frame_idx} frames...")
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video annotation complete: {frame_idx} frames processed")
    
    def _get_frame_predictions(self, frame_idx: int, start_frames: List[int],
                              classifications: List[np.ndarray], segmentations: List[np.ndarray],
                              full_signal: np.ndarray) -> Tuple[str, float, float]:
        """Get predictions for a specific frame."""
        
        # Find the closest window
        if start_frames:
            closest_idx = min(range(len(start_frames)), 
                            key=lambda i: abs(start_frames[i] - frame_idx))
            
            # Get exercise prediction
            exercise_probs = classifications[closest_idx]
            exercise_idx = np.argmax(exercise_probs)
            exercise = self.exercise_types[exercise_idx]
            exercise_prob = exercise_probs[exercise_idx]
            
            # Get rep probability
            rep_prob = full_signal[frame_idx] if frame_idx < len(full_signal) else 0.0
        else:
            exercise = "Unknown"
            exercise_prob = 0.0
            rep_prob = 0.0
        
        return exercise, exercise_prob, rep_prob
    
    def _annotate_frame(self, frame: np.ndarray, exercise: str, exercise_prob: float, 
                       rep_prob: float, rep_count: int, frame_idx: int, peaks: np.ndarray) -> None:
        """Annotate a single frame with predictions."""
        
        # Add text annotations
        cv2.putText(frame, f"Exercise: {exercise}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {exercise_prob:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Rep Prob: {rep_prob:.2f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {rep_count}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Highlight if this is a peak frame
        if frame_idx in peaks:
            cv2.putText(frame, "PEAK!", (frame.shape[1] - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


class WorkoutDemo:
    """Main demo class that orchestrates the entire process."""
    
    def __init__(self, model_path: str, window_size: int = 30, stride: int = 1):
        """Initialize the workout demo."""
        self.video_processor = VideoProcessor(model_path, window_size, stride)
        self.plot_generator = PlotGenerator(self.video_processor.exercise_types)
        self.video_annotator = VideoAnnotator(self.video_processor.exercise_types)
    
    def run_demo(self, 
                video_path: str, 
                output_dir: str, 
                windows_per_second: int = 0,
                fps: int = 30) -> Dict:
        """
        Run the complete demo pipeline.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            windows_per_second: Number of windows to process per second (0 = all)
            fps: Video frame rate
            
        Returns:
            Dictionary with processing results and file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print("="*60)
        print(f"WORKOUT TRACKER DEMO - {video_name}")
        print(f"Window Size: {self.video_processor.window_size}, Stride: {self.video_processor.stride}")
        print("="*60)
        
        # Step 1: Extract features
        features, total_frames = self.video_processor.extract_features_from_video(video_path)
        
        # Step 2: Create sliding windows
        windows = self.video_processor.create_sliding_windows(features, windows_per_second)
        
        # Step 3: Run predictions
        start_frames, classifications, segmentations = self.video_processor.predict_windows(windows)
        
        if not start_frames:
            raise ValueError("No valid predictions generated")
        
        # Step 4: Process segmentation for rep counting
        print("Processing signals for rep counting...")
        
        # Interpolate segmentation predictions to create continuous signal
        full_signal = np.zeros(total_frames)
        for i, start_frame in enumerate(start_frames):
            # Use the center frame of each window for the signal
            center_frame = start_frame + self.video_processor.window_size // 2
            if center_frame < total_frames:
                full_signal[center_frame] = segmentations[i][0] if len(segmentations[i]) > 0 else 0.0
        
        # Smooth the signal
        smoothed_signal = self.video_processor.signal_processor.smooth_signal(full_signal)
        
        # Find peaks
        peaks, _ = find_peaks(smoothed_signal, 
                            height=self.video_processor.signal_processor.min_peak_threshold,
                            distance=30)  # Minimum 1 second between peaks
        
        rep_count = len(peaks)
        
        print(f"Rep counting complete: {rep_count} repetitions detected")
        
        # Step 5: Generate outputs
        print("Generating output files...")
        
        # Generate analysis plot
        plot_path = os.path.join(output_dir, f"{video_name}_analysis.png")
        self.plot_generator.generate_analysis_plot(
            start_frames, classifications, segmentations, 
            smoothed_signal, peaks, rep_count, plot_path
        )
        
        # Generate annotated video
        video_output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
        self.video_annotator.annotate_video(
            video_path, video_output_path, start_frames, 
            classifications, segmentations, smoothed_signal, peaks, fps
        )
        
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Rep Count: {rep_count}")
        print(f"Peaks Detected: {len(peaks)}")
        print(f"Output Directory: {output_dir}")
        
        return {
            'plot_path': plot_path,
            'video_path': video_output_path
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Configurable Workout Tracker Video Demo')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', default='models/multitask_tcn_robust/best_model.keras',
                       help='Path to trained model')
    parser.add_argument('--output', default='demo_output', help='Output directory')
    parser.add_argument('--window-size', type=int, default=30, help='Window size for inference')
    parser.add_argument('--stride', type=int, default=1, help='Stride for sliding window')
    parser.add_argument('--windows_per_second', type=int, default=0,
                       help='Number of windows to process per second (0 = all windows)')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    try:
        # Create and run demo
        demo = WorkoutDemo(args.model, args.window_size, args.stride)
        results = demo.run_demo(
            video_path=args.video,
            output_dir=args.output,
            windows_per_second=args.windows_per_second,
            fps=args.fps
        )
        
        print(f"\nDemo completed successfully!")
        print(f"Check output directory: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
