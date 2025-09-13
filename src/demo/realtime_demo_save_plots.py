#!/usr/bin/env python3
"""
Multitask Real-time Demo with Saved Plots and Videos

Real-time demonstration of the multitask TCN model with 3 FPS processing.
Saves exercise classification, repetition counting, and confidence trends as plots.
Also saves annotated output videos.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import argparse
import json
from collections import deque
import time
import datetime as dt
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.dataset_builder import FeatureExtractor
from demo.signal_processor import SignalProcessor

class RealtimeDemoSavePlots:
    """Real-time demo with 3 FPS processing and saved plots and videos."""
    
    def __init__(self, model_path: str, metadata_path: str = None, target_fps: int = 3):
        """
        Initialize the real-time demo with plot and video saving.
        
        Args:
            model_path: Path to the trained multitask model
            metadata_path: Path to metadata file (optional)
            target_fps: Target processing FPS (default: 3)
        """
        print(f"Loading multitask model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata if available  
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips', 'no-exercise']
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.exercise_types = metadata.get('exercise_types', self.exercise_types)
        
        print(f"Exercise types: {self.exercise_types}")
        
        # Exercise colors for visualization
        self.exercise_colors = {
            'push-ups': (0, 255, 0),      # Green
            'squats': (255, 0, 0),        # Red  
            'pull-ups': (0, 0, 255),      # Blue
            'dips': (255, 255, 0),        # Yellow
            'no-exercise': (128, 128, 128) # Gray
        }
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Processing parameters
        self.target_fps = target_fps
        self.window_size = 15
        self.frame_skip = 30 // target_fps  # Skip frames for 3 FPS processing
        
        # State variables
        self.feature_window = deque(maxlen=self.window_size)
        self.current_exercise = "unknown"
        self.exercise_confidence = 0.0
        self.rep_probability = 0.0
        self.frame_count = 0
        self.processed_frames = 0
        self.last_process_time = 0
        
        # Initialize signal processor with improved threshold (0.5)
        self.signal_processor = SignalProcessor(rep_threshold=0.0001)
        
        # Additional state for plotting
        self.last_exercise = "unknown"
        self.segment_start_frame = 0
        self.exercise_segments = []
        
        # Plotting data
        self.plot_data = {
            'time': [],
            'frame_idx': [],
            'confidence': [],
            'rep_prob': [],
            'exercise': [],
            'exercise_names': []
        }
        
        print(f"Real-time demo initialized with {target_fps} FPS processing")
        print(f"Signal processor initialized with threshold: {self.signal_processor.rep_threshold}")
    
    def should_process_frame(self, current_time: float) -> bool:
        """Check if frame should be processed based on target FPS."""
        time_since_last = current_time - self.last_process_time
        return time_since_last >= (1.0 / self.target_fps)
    
    def predict_window(self, features: np.ndarray) -> Tuple[str, float, float]:
        """
        Predict exercise and repetition probability for a window of features.
        
        Args:
            features: Window of features (window_size, feature_dim)
            
        Returns:
            Tuple of (exercise, confidence, rep_probability)
        """
        # Reshape for model input
        X = features.reshape(1, self.window_size, -1)
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        cls_pred, seg_pred = predictions
        
        # Get exercise classification
        exercise_idx = np.argmax(cls_pred[0])
        confidence = float(np.max(cls_pred[0]))
        exercise = self.exercise_types[exercise_idx]
        
        # Get average repetition probability
        rep_prob = float(np.mean(seg_pred[0]))
        
        return exercise, confidence, rep_prob
    
    def detect_rep(self, exercise: str, rep_prob: float, frame_idx: int):
        """Detect repetitions using the signal processor."""
        if exercise == "no-exercise":
            return
        
        # Use the signal processor for rep detection
        rep_detected = self.signal_processor.detect_rep(exercise, rep_prob, frame_idx)
        
        # Track exercise segments for plotting
        if exercise != self.last_exercise:
            if self.last_exercise != "unknown":
                # End previous segment
                self.exercise_segments.append({
                    'exercise': self.last_exercise,
                    'start_frame': self.segment_start_frame,
                    'end_frame': frame_idx - 1
                })
            
            # Start new segment
            self.segment_start_frame = frame_idx
            self.last_exercise = exercise
    
    def annotate_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Add annotations to the frame for video output."""
        height, width = frame.shape[:2]
        annotated = frame.copy()
        
        # Exercise info
        exercise_text = f"Exercise: {self.current_exercise}"
        confidence_text = f"Confidence: {self.exercise_confidence:.3f}"
        rep_prob_text = f"Rep Prob: {self.rep_probability:.3f}"
        
        # Rep counts for all exercises
        rep_texts = []
        for exercise in self.exercise_types:
            if exercise != "no-exercise":
                count = self.signal_processor.get_rep_count(exercise)
                if count > 0:
                    rep_texts.append(f"{exercise}: {count}")
        
        # Performance info
        fps_text = f"Processing FPS: {self.target_fps}"
        frame_text = f"Frame: {frame_idx}"
        threshold_text = f"Threshold: {self.signal_processor.rep_threshold}"
        
        # Draw text
        y_offset = 30
        cv2.putText(annotated, exercise_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(annotated, confidence_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(annotated, rep_prob_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Rep counts
        for rep_text in rep_texts:
            cv2.putText(annotated, rep_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        # Performance info
        cv2.putText(annotated, fps_text, (10, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, frame_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, threshold_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Progress bar
        progress = (frame_idx / self.total_frames) * 100 if hasattr(self, 'total_frames') else 0
        progress_text = f"Progress: {progress:.1f}%"
        cv2.putText(annotated, progress_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated
    
    def process_video(self, video_path: str, output_dir: str = "output_videos"):
        """
        Process a video file and save plots and videos.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output plots and videos
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.total_frames = total_frames
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Frames: {total_frames}, Duration: {duration:.1f}s")
        print(f"Resolution: {width}x{height}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video name for output files
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, f'{video_name}_output.mp4')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Extract features from every frame
                features = self.feature_extractor.extract_angles(frame)
                if features is not None:
                    self.feature_window.append(features)
                
                # Only run model inference at target FPS intervals
                if (len(self.feature_window) == self.window_size and 
                    self.should_process_frame(current_time)):
                    
                    # Convert window to numpy array
                    features_array = np.array(list(self.feature_window))
                    
                    # Get predictions
                    exercise, confidence, rep_prob = self.predict_window(features_array)
                    
                    # Update state
                    self.current_exercise = exercise
                    self.exercise_confidence = confidence
                    self.rep_probability = rep_prob
                    
                    # Detect repetitions
                    self.detect_rep(exercise, rep_prob, frame_idx)
                    
                    # Store data for plotting
                    self.plot_data['time'].append(current_time - start_time)
                    self.plot_data['frame_idx'].append(frame_idx)
                    self.plot_data['confidence'].append(confidence)
                    self.plot_data['rep_prob'].append(rep_prob)
                    self.plot_data['exercise'].append(exercise)
                    self.plot_data['exercise_names'].append(self.exercise_types.index(exercise))
                    
                    # Update processing time
                    self.last_process_time = current_time
                    self.processed_frames += 1
                
                # Annotate frame for video output
                annotated_frame = self.annotate_frame(frame, frame_idx)
                
                # Write frame to output video
                writer.write(annotated_frame)
                
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            cap.release()
            writer.release()
            
            # End final segment
            if self.last_exercise != "unknown":
                self.exercise_segments.append({
                    'exercise': self.last_exercise,
                    'start_frame': self.segment_start_frame,
                    'end_frame': frame_idx - 1
                })
            
            # Save plots
            self.save_plots(output_dir, video_name)
            
            # Print final statistics
            print(f"\nProcessing completed!")
            print(f"Total frames: {frame_idx}")
            print(f"Processed frames: {self.processed_frames}")
            print(f"Output video saved: {output_video_path}")
            print(f"Final rep counts:")
            for exercise in self.exercise_types:
                if exercise != "no-exercise":
                    count = self.signal_processor.get_rep_count(exercise)
                    print(f"  {exercise}: {count}")
    
    def save_plots(self, output_dir: str, video_name: str):
        """Save comprehensive plots of the analysis."""
        if not self.plot_data['time']:
            print("No data to plot")
            return
        
        # Convert to numpy arrays
        times = np.array(self.plot_data['time'])
        frames = np.array(self.plot_data['frame_idx'])
        confidences = np.array(self.plot_data['confidence'])
        rep_probs = np.array(self.plot_data['rep_prob'])
        exercises = np.array(self.plot_data['exercise'])
        exercise_names = np.array(self.plot_data['exercise_names'])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Workout Analysis: {video_name}', fontsize=16)
        
        # Plot 1: Exercise Classification over Time
        ax1.plot(times, exercise_names, 'o-', markersize=3, linewidth=1)
        ax1.set_ylabel('Exercise Type')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Exercise Classification Over Time')
        ax1.set_yticks(range(len(self.exercise_types)))
        ax1.set_yticklabels(self.exercise_types)
        ax1.grid(True, alpha=0.3)
        
        # Add exercise segments as background
        for segment in self.exercise_segments:
            start_time = times[segment['start_frame']] if segment['start_frame'] < len(times) else 0
            end_time = times[min(segment['end_frame'], len(times)-1)] if segment['end_frame'] < len(times) else times[-1]
            exercise_idx = self.exercise_types.index(segment['exercise'])
            ax1.axhspan(exercise_idx - 0.4, exercise_idx + 0.4, 
                       xmin=start_time/times[-1], xmax=end_time/times[-1], 
                       alpha=0.2, color=f'C{exercise_idx}')
        
        # Plot 2: Repetition Probability and Threshold
        ax2.plot(times, rep_probs, 'b-', linewidth=1, label='Rep Probability')
        ax2.axhline(y=self.signal_processor.rep_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax2.set_ylabel('Repetition Probability')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Repetition Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight detected peaks with better visualization
        for exercise in self.exercise_types:
            if exercise != "no-exercise":
                peaks = self.signal_processor.get_all_peaks(exercise)
                if peaks:
                    peak_times = []
                    peak_probs = []
                    for frame, prob in peaks:
                        # Find the closest time index for this frame
                        if frame < len(times):
                            peak_times.append(times[frame])
                            peak_probs.append(prob)
                    
                    if peak_times:
                        ax2.scatter(peak_times, peak_probs, color='red', s=100, alpha=0.8, 
                                   zorder=5, marker='^', label=f'{exercise} peaks')
                        # Add vertical lines to show peak locations
                        for pt, pp in zip(peak_times, peak_probs):
                            ax2.axvline(x=pt, color='red', alpha=0.3, linestyle='-')

        # Plot 3: Confidence Scores
        ax3.plot(times, confidences, 'g-', linewidth=1, label='Confidence')
        ax3.set_ylabel('Confidence Score')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Classification Confidence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Rep Counts by Exercise
        exercise_counts = {}
        for exercise in self.exercise_types:
            if exercise != "no-exercise":
                exercise_counts[exercise] = self.signal_processor.get_rep_count(exercise)
        
        if exercise_counts:
            exercises_list = list(exercise_counts.keys())
            counts_list = list(exercise_counts.values())
            colors = [self.exercise_colors.get(ex, 'gray') for ex in exercises_list]
            
            bars = ax4.bar(exercises_list, counts_list, color=[np.array(c)/255.0 for c in colors])
            ax4.set_ylabel('Repetition Count')
            ax4.set_xlabel('Exercise Type')
            ax4.set_title('Total Repetitions by Exercise')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts_list):
                if count > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{video_name}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved: {plot_path}")
        
        # Save detailed data
        data_path = os.path.join(output_dir, f'{video_name}_data.json')
        analysis_data = {
            'video_name': video_name,
            'total_frames': len(frames),
            'processed_frames': self.processed_frames,
            'duration_seconds': times[-1] if times.size > 0 else 0,
            'rep_counts': {ex: self.signal_processor.get_rep_count(ex) for ex in self.exercise_types if ex != "no-exercise"},
            'threshold': self.signal_processor.rep_threshold,
            'exercise_segments': self.exercise_segments,
            'timestamps': times.tolist(),
            'confidences': confidences.tolist(),
            'rep_probabilities': rep_probs.tolist(),
            'exercises': exercises.tolist()
        }
        
        with open(data_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"Analysis data saved: {data_path}")
        
        plt.show()


def main():
    """Main function for real-time demo with plots and videos."""
    parser = argparse.ArgumentParser(description='Multitask Real-time Demo with Plots and Videos')
    parser.add_argument('--model', required=True, help='Path to multitask model')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--metadata', help='Path to metadata file')
    parser.add_argument('--output', default='output_videos', help='Output directory for plots and videos')
    parser.add_argument('--fps', type=int, default=3, help='Target processing FPS')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], default='high', 
                       help='Output quality')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = RealtimeDemoSavePlots(
        model_path=args.model,
        metadata_path=args.metadata,
        target_fps=args.fps
    )
    
    # Process video
    demo.process_video(args.video, args.output)


if __name__ == "__main__":
    main()
