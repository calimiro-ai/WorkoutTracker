#!/usr/bin/env python3
"""
Multitask Camera Demo

Real-time camera demonstration of the multitask TCN model.
Processes live camera feed with 3 FPS processing for optimal performance.
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.dataset_builder import FeatureExtractor
from demo.signal_processor import SignalProcessor

class CameraDemo:
    """Real-time camera demo with 3 FPS processing for optimal performance."""
    
    def __init__(self, model_path: str, metadata_path: str = None, target_fps: int = 3):
        """
        Initialize the camera demo.
        
        Args:
            model_path: Path to the trained multitask model
            metadata_path: Path to metadata file (optional)
            target_fps: Target processing FPS (default: 3)
        """
        print(f"Loading multitask model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata if available  
        self.exercise_types = ['dips', 'pull-ups', 'push-ups', 'squats', 'no-exercise']
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.exercise_types = metadata.get('exercise_types', self.exercise_types)
        
        print(f"Exercise types: {self.exercise_types}")
        
        # Exercise colors for visualization
        self.exercise_colors = {
            'dips': (255, 100, 100),        # Light Red
            'pull-ups': (100, 255, 100),    # Light Green  
            'push-ups': (100, 100, 255),    # Light Blue
            'squats': (255, 255, 100),      # Light Yellow
            'no-exercise': (180, 180, 180), # Light Gray
            'unknown': (128, 128, 128)      # Gray
        }
        
        # Initialize feature extractor and state
        self.feature_extractor = FeatureExtractor()
        self.window_size = 30
        self.feature_window = deque(maxlen=self.window_size)
        
        # Real-time processing parameters
        self.target_fps = target_fps
        self.last_process_time = 0
        self.process_interval = 1.0 / target_fps  # Time between processing calls
        
        # Initialize signal processor with improved threshold (0.5)
        self.signal_processor = SignalProcessor()
        
        
        # State tracking
        self.reset_state()
        
        print(f"Camera demo initialized! Target FPS: {target_fps}")
        print(f"Signal processor initialized with threshold: {self.signal_processor.rep_threshold}")
    
    def reset_state(self):
        """Reset all tracking state."""
        self.frame_count = 0
        self.processed_frames = 0
        self.current_exercise = "no-exercise"
        self.current_confidence = 0.0
        self.current_rep_prob = 0.0
        
        # Reset signal processor
        self.signal_processor.reset_counts()
        
        print("State reset - all counts cleared!")
    
    def should_process_frame(self, current_time: float) -> bool:
        """Check if enough time has passed to process the next frame."""
        time_since_last = current_time - self.last_process_time
        return time_since_last >= self.process_interval
    
    def predict_window(self, features_window: np.ndarray) -> Tuple[str, float, float]:
        """
        Predict exercise and repetition probability for a window of features.
        
        Args:
            features_window: Window of features (window_size, feature_dim)
            
        Returns:
            Tuple of (exercise, confidence, rep_probability)
        """
        # Reshape for model input
        X = features_window.reshape(1, self.window_size, -1)
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        cls_pred, seg_pred = predictions
        
        # Get exercise classification
        exercise_idx = np.argmax(cls_pred[0])
        confidence = float(np.max(cls_pred[0]))
        exercise = self.exercise_types[exercise_idx]
        
        # Get average repetition probability
        rep_probability = float(np.mean(seg_pred[0]))
        
        return exercise, confidence, rep_probability
    
    def detect_rep(self, exercise: str, rep_probability: float, frame_idx: int) -> bool:
        """
        Detect repetition using the signal processor.
        
        Args:
            exercise: Current exercise type
            rep_probability: Current rep probability
            frame_idx: Current frame index
            
        Returns:
            True if a rep was detected
        """
        return self.signal_processor.detect_rep_advanced(exercise, rep_probability, frame_idx)
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process a single frame for exercise detection and rep counting.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Annotated frame
        """
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
            
            # Update current state
            self.current_exercise = exercise
            self.current_confidence = confidence
            self.current_rep_prob = rep_prob
            
            # Detect repetitions
            rep_detected = self.detect_rep(exercise, rep_prob, frame_idx)
            
            # Update processing time
            self.last_process_time = current_time
            self.processed_frames += 1
        
        # Create annotated frame
        annotated_frame = self.annotate_frame(frame, frame_idx)
        
        return annotated_frame
    
    def annotate_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Add annotations to the frame."""
        height, width = frame.shape[:2]
        annotated = frame.copy()
        
        # Exercise info
        exercise_text = f"Exercise: {self.current_exercise}"
        confidence_text = f"Confidence: {self.current_confidence:.3f}"
        rep_prob_text = f"Rep Prob: {self.current_rep_prob:.3f}"
        
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
        cv2.putText(annotated, fps_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, frame_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Threshold indicator
        threshold_text = f"Threshold: {self.signal_processor.rep_threshold}"
        cv2.putText(annotated, threshold_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated
    
    def run_camera(self, camera_id: int = 0, save_video: bool = False, output_path: str = None):
        """
        Run the camera demo.
        
        Args:
            camera_id: Camera device ID
            save_video: Whether to save video output
            output_path: Output video path
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer if saving
        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        
        print("Camera demo started! Press 'q' to quit, 'r' to reset counts")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame, self.frame_count)
                
                # Save frame if writer is available
                if writer is not None:
                    writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Workout Tracker - Camera Demo', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_state()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\nDemo completed!")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Model inference frames: {self.processed_frames}")
            print(f"Final rep counts:")
            for exercise in self.exercise_types:
                if exercise != "no-exercise":
                    count = self.signal_processor.get_rep_count(exercise)
                    print(f"  {exercise}: {count}")


def main():
    """Main function for camera demo."""
    parser = argparse.ArgumentParser(description='Multitask Camera Demo')
    parser.add_argument('--model', required=True, help='Path to multitask model')
    parser.add_argument('--metadata', help='Path to metadata file')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--fps', type=int, default=3, help='Target processing FPS')
    parser.add_argument('--save', action='store_true', help='Save video output')
    parser.add_argument('--output', help='Output video path')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = CameraDemo(
        model_path=args.model,
        metadata_path=args.metadata,
        target_fps=args.fps
    )
    
    # Run camera demo
    demo.run_camera(
        camera_id=args.camera,
        save_video=args.save,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
