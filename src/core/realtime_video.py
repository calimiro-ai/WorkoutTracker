#!/usr/bin/env python3
"""
Real-time video file exercise detection with live rep counting using peak detection
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
import threading
from collections import deque
from typing import Optional, Dict, List, Tuple
from queue import Queue
import csv
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.dataset_builder import FeatureExtractor
from core.realtime_pipeline import ExerciseClassifier
from core.frontend_interface import update_workout_state

# --- RealTimePeakDetector and SegmentationProcessor copied from realtime.py ---

class RealTimePeakDetector:
    """Real-time peak detection for exercise repetition counting."""
    def __init__(self, min_peak_distance: int = 15, min_threshold: float = 0.6):
        self.min_peak_distance = min_peak_distance
        self.min_threshold = min_threshold
        self.probability_history = deque(maxlen=5)
        self.frame_history = deque(maxlen=5)
        self.rep_count = 0
        self.last_peak_frame = -min_peak_distance
        self.last_probability = 0.0
        self.rising = False
        self.peak_candidate = None
        self.last_rep_time = time.time()
        self.rep_times = deque(maxlen=10)
    def update(self, frame_idx: int, probability: float) -> bool:
        self.probability_history.append(probability)
        self.frame_history.append(frame_idx)
        if len(self.probability_history) < 2:
            self.last_probability = probability
            return False
        peak_detected = False
        if probability > self.last_probability:
            if not self.rising:
                self.rising = True
            self.peak_candidate = (frame_idx, probability)
        elif self.rising and probability < self.last_probability:
            if self.peak_candidate is not None:
                peak_frame, peak_prob = self.peak_candidate
                if (frame_idx - self.last_peak_frame >= self.min_peak_distance and 
                    peak_prob >= self.min_threshold):
                    self.rep_count += 1
                    self.last_peak_frame = peak_frame
                    peak_detected = True
                    current_time = time.time()
                    if self.last_rep_time > 0:
                        rep_time = current_time - self.last_rep_time
                        self.rep_times.append(rep_time)
                    self.last_rep_time = current_time
                self.peak_candidate = None
            self.rising = False
        self.last_probability = probability
        return peak_detected
    def get_rep_count(self) -> int:
        return self.rep_count
    def get_avg_rep_time(self) -> float:
        if len(self.rep_times) > 0:
            return np.mean(self.rep_times)
        return 0.0
    def reset(self):
        self.probability_history.clear()
        self.frame_history.clear()
        self.rep_count = 0
        self.last_peak_frame = -self.min_peak_distance
        self.last_probability = 0.0
        self.rising = False
        self.peak_candidate = None
        self.last_rep_time = time.time()
        self.rep_times.clear()

class SegmentationProcessor:
    """Real-time segmentation processor for exercise repetition detection."""
    def __init__(self, models_dir: str = "models/segmentation"):
        self.models_dir = models_dir
        self.models = {}
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips']
        for exercise in self.exercise_types:
            model_path = os.path.join(models_dir, f"{exercise}.keras")
            if os.path.exists(model_path):
                self.models[exercise] = tf.keras.models.load_model(model_path, compile=False)
                print(f"Loaded {exercise} segmentation model")
            else:
                print(f"Warning: {exercise} segmentation model not found at {model_path}")
        self.csv_file = open("realtime_segmentation_probabilities.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "timestamp", "exercise", "probability"])
        self.frame_counter = 0
    def predict_window(self, window: np.ndarray, exercise_type: str) -> float:
        if exercise_type not in self.models:
            return 0.0
        sequence = window[None, ...]
        probability = self.models[exercise_type].predict(sequence, verbose=0)
        prob_value = float(probability[0, -1, 0])
        self.frame_counter += 1
        self.csv_writer.writerow([self.frame_counter, time.time(), exercise_type, prob_value])
        if self.frame_counter % 10 == 0:
            self.csv_file.flush()
        return prob_value
    def close(self):
        self.csv_file.close()

class VideoRealtimeWithRepsPipeline:
    """Real-time video file exercise detection with live rep counting."""
    def __init__(self, 
                 video_path: str,
                 classifier_model: str = "models/classification/exercise_classifier.keras",
                 segmentation_models_dir: str = "models/segmentation",
                 window_size: int = 30):
        self.video_path = video_path
        self.classifier = ExerciseClassifier(classifier_model, window_size)
        self.segmentation = SegmentationProcessor(segmentation_models_dir)
        self.feature_extractor = FeatureExtractor()
        self.window_size = window_size
        self.peak_detector = RealTimePeakDetector()
        self.rep_counters = {
            'push-ups': RealTimePeakDetector(),
            'squats': RealTimePeakDetector(),
            'pull-ups': RealTimePeakDetector(),
            'dips': RealTimePeakDetector()
        }
        self.current_window = np.zeros((window_size, 25), dtype=np.float32)
        self.window_filled = False
        self.current_exercise = "unknown"
        self.exercise_confidence = 0.0
        self.current_probability = 0.0
        self.frame_count = 0
        self.processed_frame_count = 0
        self.feature_queue = Queue(maxsize=100)
        self.result_queue = Queue(maxsize=10)
        self.running = True
        self.processing_times = deque(maxlen=30)
        self.fps_times = deque(maxlen=30)
        self.exercise_predictions = []
        self.confidence_scores = []
        self.probability_scores = []
        print(f"Video real-time pipeline with rep counting initialized for video: {video_path}")
        print(f"Press 'q' to quit, 'r' to reset counts")
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties: {width}x{height}, {fps} FPS")
        processing_thread = threading.Thread(target=self._background_processor)
        processing_thread.start()
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        last_fps_time = time.time()
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                features = self.feature_extractor.extract_angles(frame)
                if features is not None:
                    try:
                        self.feature_queue.put_nowait((self.frame_count, features))
                    except:
                        pass
                if features is not None:
                    self.current_window = np.roll(self.current_window, -1, axis=0)
                    self.current_window[-1] = features
                    if not self.window_filled and self.frame_count >= self.window_size - 1:
                        self.window_filled = True
                try:
                    while not self.result_queue.empty():
                        exercise, confidence, probability = self.result_queue.get_nowait()
                        if exercise != self.current_exercise:
                            self._update_frontend_state()
                        self.current_exercise = exercise
                        self.exercise_confidence = confidence
                        self.current_probability = probability
                        if exercise in self.rep_counters:
                            rep_detected = self.rep_counters[exercise].update(self.frame_count, probability)
                            if rep_detected:
                                print(f"REP DETECTED! {exercise}: {self.rep_counters[exercise].get_rep_count()}")
                                self._update_frontend_state()
                except:
                    pass
                self.exercise_predictions.append(self.current_exercise)
                self.confidence_scores.append(self.exercise_confidence)
                self.probability_scores.append(self.current_probability)
                current_time = time.time()
                self.fps_times.append(current_time - last_fps_time)
                last_fps_time = current_time
                self._draw_overlay(frame)
                cv2.imshow('Real-Time Video Detection with Rep Counting', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_all_counts()
                self.frame_count += 1
                processing_time = time.time() - frame_start_time
                if processing_time < frame_delay:
                    time.sleep(frame_delay - processing_time)
        finally:
            self.running = False
            processing_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            self.segmentation.close()
        self._print_results()
        self._plot_probabilities()
    def _background_processor(self):
        while self.running:
            try:
                frame_idx, features = self.feature_queue.get(timeout=0.1)
                self.current_window = np.roll(self.current_window, -1, axis=0)
                self.current_window[-1] = features
                if not self.window_filled and frame_idx >= self.window_size - 1:
                    self.window_filled = True
                if self.window_filled and frame_idx % 10 == 0:
                    start_time = time.time()
                    window = self.current_window.copy()
                    exercise, confidence = self.classifier.predict_window(window)
                    probability = 0.0
                    if exercise != "unknown":
                        probability = self.segmentation.predict_window(window, exercise)
                    try:
                        self.result_queue.put_nowait((exercise, confidence, probability))
                    except:
                        pass
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    self.processed_frame_count += 1
            except:
                continue
    def _draw_overlay(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        exercise_text = f"Exercise: {self.current_exercise.upper()}"
        confidence_text = f"Confidence: {self.exercise_confidence:.3f}"
        probability_text = f"Rep Probability: {self.current_probability:.3f}"
        if self.exercise_confidence > 0.7:
            color = (0, 255, 0)
        elif self.exercise_confidence > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.putText(frame, exercise_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.current_probability > 0.7:
            prob_color = (0, 255, 0)
        elif self.current_probability > 0.5:
            prob_color = (0, 255, 255)
        else:
            prob_color = (0, 0, 255)
        cv2.putText(frame, probability_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 2)
        if self.current_exercise in self.rep_counters:
            counter = self.rep_counters[self.current_exercise]
            rep_count = counter.get_rep_count()
            avg_rep_time = counter.get_avg_rep_time()
            rep_text = f"Reps: {rep_count}"
            cv2.putText(frame, rep_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if avg_rep_time > 0:
                time_text = f"Avg Time: {avg_rep_time:.1f}s"
                cv2.putText(frame, time_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            cv2.putText(frame, f"Model FPS: {1000/avg_time:.1f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.fps_times:
            video_fps = 1.0 / np.mean(self.fps_times)
            cv2.putText(frame, f"Video FPS: {video_fps:.1f}", (width - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    def _reset_all_counts(self):
        for counter in self.rep_counters.values():
            counter.reset()
        print("All rep counts reset!")
        self._update_frontend_state()
    def _update_frontend_state(self):
        total_reps = {exercise: counter.get_rep_count() for exercise, counter in self.rep_counters.items()}
        update_workout_state(total_reps, self.current_exercise)
    def _print_results(self):
        print("\n" + "="*50)
        print("VIDEO REAL-TIME PROCESSING RESULTS WITH REP COUNTING")
        print("="*50)
        print(f"\nExercise Classification:")
        print(f"Total frames processed: {len(self.exercise_predictions)}")
        from collections import Counter
        exercise_counts = Counter(self.exercise_predictions)
        for exercise, count in exercise_counts.most_common():
            percentage = count / len(self.exercise_predictions) * 100
            print(f"  {exercise}: {count} frames ({percentage:.1f}%)")
        print(f"\nRep Counting Results:")
        for exercise, counter in self.rep_counters.items():
            rep_count = counter.get_rep_count()
            avg_time = counter.get_avg_rep_time()
            print(f"  {exercise}: {rep_count} reps")
            if avg_time > 0:
                print(f"    Average time between reps: {avg_time:.2f}s")
        if self.confidence_scores:
            avg_confidence = np.mean(self.confidence_scores)
            max_confidence = np.max(self.confidence_scores)
            min_confidence = np.min(self.confidence_scores)
            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_confidence:.3f}")
            print(f"  Maximum: {max_confidence:.3f}")
            print(f"  Minimum: {min_confidence:.3f}")
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            min_time = np.min(self.processing_times) * 1000
            print(f"\nPerformance Statistics:")
            print(f"  Model inference time: {avg_time:.1f}ms")
            print(f"  Maximum inference time: {max_time:.1f}ms")
            print(f"  Minimum inference time: {min_time:.1f}ms")
            print(f"  Model FPS: {1000/avg_time:.1f}")
            print(f"  Frames analyzed: {self.processed_frame_count}")
    def _plot_probabilities(self):
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.read_csv("realtime_segmentation_probabilities.csv")
            if len(df) == 0:
                print("No segmentation data to plot")
                return
            plt.figure(figsize=(15, 8))
            for exercise in self.segmentation.exercise_types:
                exercise_data = df[df['exercise'] == exercise]
                if len(exercise_data) > 0:
                    plt.plot(exercise_data['frame'], exercise_data['probability'], \
                            label=exercise, linewidth=2, alpha=0.8)
            plt.xlabel('Frame Number')
            plt.ylabel('Repetition Probability')
            plt.title('Real-Time Exercise Repetition Detection Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig('realtime_segmentation_probabilities_plot.png', dpi=300, bbox_inches='tight')
            print(f"Segmentation probabilities plot saved to: realtime_segmentation_probabilities_plot.png")
            plt.show()
        except ImportError:
            print("Matplotlib not available, skipping probability plot")
        except Exception as e:
            print(f"Error plotting probabilities: {e}")

def main():
    parser = argparse.ArgumentParser(description='Real-time video file exercise detection with rep counting')
    parser.add_argument('--input', type=str, required=True, help='Path to the video file to process')
    parser.add_argument('--classifier', type=str, 
                       default='models/classification/exercise_classifier.keras',
                       help='Path to classification model')
    parser.add_argument('--models-dir', type=str, 
                       default='models/segmentation',
                       help='Directory containing segmentation models')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size for classification')
    args = parser.parse_args()
    pipeline = VideoRealtimeWithRepsPipeline(
        video_path=args.input,
        classifier_model=args.classifier,
        segmentation_models_dir=args.models_dir,
        window_size=args.window_size
    )
    pipeline.run()

if __name__ == '__main__':
    main() 