#!/usr/bin/env python3
"""
Live Workout Tracker Demo

Real-time workout tracking using live camera feed.
- Captures live camera stream
- Processes windows at 3 FPS (3 times per second)
- Detects peaks and counts repetitions
- Uses delay mechanism to confirm peaks (2 additional inferences)
- Real-time visualization with rep counter
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.dataset_builder import FeatureExtractor
from training.model import build_multitask_tcn_model


class LiveWorkoutTracker:
    """Live workout tracker with real-time inference and rep counting."""
    
    def __init__(self, model_path: str, window_size: int = 30, inference_rate: int = 3, 
                 peak_delay: int = 2, min_peak_height: float = 0.3):
        """
        Initialize the live workout tracker.
        
        Args:
            model_path: Path to trained model
            window_size: Size of sliding window for inference
            inference_rate: How many times per second to run inference
            peak_delay: Number of additional inferences to wait before confirming peak
            min_peak_height: Minimum height for peak detection
        """
        self.model_path = model_path
        self.window_size = window_size
        self.inference_rate = inference_rate
        self.peak_delay = peak_delay
        self.min_peak_height = min_peak_height
        
        # Initialize components
        self.extractor = FeatureExtractor()
        self.model = None
        self.cap = None
        
        # Data storage
        self.frame_buffer = deque(maxlen=window_size)
        self.feature_buffer = deque(maxlen=window_size)
        self.rep_signal = deque(maxlen=100)  # Store last 100 rep probabilities
        self.peak_history = deque(maxlen=10)  # Store recent peaks
        
        # State variables
        self.rep_count = 0
        self.last_peak_time = 0
        self.is_running = False
        self.current_exercise = "Unknown"
        
        # Threading
        self.inference_thread = None
        self.stop_event = threading.Event()
        
        print(f"Live workout tracker initialized:")
        print(f"  Model: {model_path}")
        print(f"  Window size: {window_size}")
        print(f"  Inference Rate: {inference_rate}")
        print(f"  Peak delay: {peak_delay} inferences")
        print(f"  Min peak height: {min_peak_height}")
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def start_camera(self, camera_id: int = 0):
        """Start camera capture."""
        print(f"Starting camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera started successfully!")
        return True
    
    def process_frame(self, frame):
        """Process a single frame and extract features."""
        try:
            # Extract pose features
            features = self.extractor.extract_angles(frame)
            if features is not None:
                self.feature_buffer.append(features)
                return True
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return False
    
    def run_inference(self):
        """Run inference on current window."""
        if len(self.feature_buffer) < self.window_size:
            return None
        
        try:
            # Get the most recent window
            window_features = np.array(list(self.feature_buffer)[-self.window_size:])
            window_features = window_features.reshape(1, self.window_size, -1)
            
            # Run inference
            predictions = self.model.predict(window_features, verbose=0)
            
            # Extract classification and segmentation
            classification_probs = predictions[0][0]  # Classification probabilities
            segmentation_probs = predictions[1][0]  # Segmentation probabilities
            
            # Get exercise type
            exercise_idx = np.argmax(classification_probs)
            exercise_names = ['push-ups', 'squats', 'pull-ups', 'dips']
            self.current_exercise = exercise_names[exercise_idx] if exercise_idx < len(exercise_names) else "Unknown"
            
            # Get rep probability (max of segmentation)
            rep_prob = np.max(segmentation_probs)
            
            return {
                'exercise': self.current_exercise,
                'exercise_confidence': float(np.max(classification_probs)),
                'rep_probability': float(rep_prob),
                'segmentation': segmentation_probs
            }
            
        except Exception as e:
            print(f"Error in inference: {e}")
            return None
    
    def detect_peaks(self):
        """Detect peaks in the rep signal with delay mechanism."""
        if len(self.rep_signal) < 10:  # Need some data
            return False
        
        # Convert to numpy array
        signal = np.array(list(self.rep_signal))
        
        # Smooth the signal
        smoothed_signal = gaussian_filter1d(signal, sigma=1.0)
        
        # Find peaks
        peaks, properties = find_peaks(
            smoothed_signal, 
            height=self.min_peak_height,
            distance=5,  # Minimum distance between peaks
            prominence=0.1
        )
        
        if len(peaks) > 0:
            # Get the most recent peak
            latest_peak_idx = peaks[-1]
            latest_peak_time = len(self.rep_signal) - 1 - latest_peak_idx
            
            # Check if this is a new peak (not already counted)
            if latest_peak_time not in self.peak_history:
                # Apply delay mechanism: wait for additional inferences
                if len(self.rep_signal) >= latest_peak_idx + self.peak_delay:
                    # Check if signal is still high after delay
                    future_values = list(self.rep_signal)[-self.peak_delay:]
                    if all(v > self.min_peak_height * 0.8 for v in future_values):
                        # Confirm peak
                        self.rep_count += 1
                        self.peak_history.append(latest_peak_time)
                        self.last_peak_time = time.time()
                        return True
        
        return False
    
    def inference_loop(self):
        """Main inference loop running in separate thread."""
        inference_interval = 1.0 / self.inference_rate
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            # Run inference if we have enough data
            if len(self.feature_buffer) >= self.window_size:
                result = self.run_inference()
                
                if result is not None:
                    # Add to rep signal
                    self.rep_signal.append(result['rep_probability'])
                    
                    # Detect peaks
                    peak_detected = self.detect_peaks()
                    
                    # Update display info
                    self.latest_result = result
                    if peak_detected:
                        print(f"Rep detected! Total: {self.rep_count}")
            
            # Sleep for the remaining time
            elapsed = time.time() - start_time
            sleep_time = max(0, inference_interval - elapsed)
            time.sleep(sleep_time)
    
    def draw_overlay(self, frame):
        """Draw overlay information on frame."""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Background for text
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Rep counter (large and prominent)
        rep_text = f"Reps: {self.rep_count}"
        cv2.putText(frame, rep_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Exercise type
        exercise_text = f"Exercise: {self.current_exercise}"
        cv2.putText(frame, exercise_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Confidence
        if hasattr(self, 'latest_result'):
            conf_text = f"Confidence: {self.latest_result['exercise_confidence']:.2f}"
            cv2.putText(frame, conf_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Rep probability
        if hasattr(self, 'latest_result'):
            prob_text = f"Rep Prob: {self.latest_result['rep_probability']:.3f}"
            cv2.putText(frame, prob_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        inst_text = "Press 'q' to quit, 'r' to reset counter"
        cv2.putText(frame, inst_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id: int = 0):
        """Run the live workout tracker."""
        print("Starting live workout tracker...")
        
        # Load model
        if not self.load_model():
            return False
        
        # Start camera
        if not self.start_camera(camera_id):
            return False
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.is_running = True
        print("Live tracking started! Press 'q' to quit, 'r' to reset counter")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame for features
                self.process_frame(frame)
                
                # Draw overlay
                display_frame = self.draw_overlay(frame)
                
                # Show frame
                cv2.imshow('Live Workout Tracker', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("Resetting counter...")
                    self.rep_count = 0
                    self.rep_signal.clear()
                    self.peak_history.clear()
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.is_running = False
        self.stop_event.set()
        
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Live Workout Tracker Demo')
    parser.add_argument('--model', default='models/final/best_model.keras',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--window-size', type=int, default=30, help='Window size for inference')
    parser.add_argument('--inference-rate', type=int, default=3, help='Inference frequency (times per second)')
    parser.add_argument('--peak-delay', type=int, default=2, help='Peak confirmation delay (inferences)')
    parser.add_argument('--min-peak', type=float, default=0.3, help='Minimum peak height')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    try:
        # Create and run tracker
        tracker = LiveWorkoutTracker(
            model_path=args.model,
            window_size=args.window_size,
            inference_rate=args.inference_rate,
            peak_delay=args.peak_delay,
            min_peak_height=args.min_peak
        )
        
        tracker.run(camera_id=args.camera)
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
