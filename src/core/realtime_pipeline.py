#!/usr/bin/env python3
"""
Real-Time Exercise Detection & Rep Counting Pipeline.

This module provides a complete real-time pipeline that:
1. Captures video from ESP32-CAM or webcam
2. Classifies the current exercise type
3. Counts repetitions using exercise-specific segmentation models
4. Displays results in real-time with minimal latency
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
from typing import Optional, Dict, List, Tuple
import argparse
import os
import csv
import sys
import os
sys.path.append(os.path.dirname(__file__))
from dataset_builder import FeatureExtractor

class ExerciseClassifier:
    """Real-time exercise classifier using TCN model."""
    def __init__(self, model_path: str, window_size: int = 30):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.window_size = window_size
        self.feature_buffer = deque(maxlen=window_size)
        self.extractor = FeatureExtractor()
        # Load class names
        class_names_path = model_path.replace('.keras', '_classes.txt')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = ['dips', 'pull-ups', 'push-ups', 'squats']
        print(f"Loaded classifier with {len(self.class_names)} classes: {self.class_names}")

    def update(self, frame: np.ndarray):
        features = self.extractor.extract_angles(frame)
        if features is None:
            features = np.zeros(25, dtype=np.float32)
        self.feature_buffer.append(features)
        if len(self.feature_buffer) < self.window_size:
            return "unknown", 0.0
        sequence = np.array(list(self.feature_buffer))[None, ...]
        probabilities = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(probabilities[0])
        confidence = float(probabilities[0][predicted_class])
        return self.class_names[predicted_class], confidence

    def predict_window(self, window: np.ndarray):
        X = window[None, ...]
        preds = self.model.predict(X, verbose=0)
        pred = preds[0]
        class_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        class_name = self.class_names[class_idx] if confidence > 0.5 else "unknown"
        return class_name, confidence

class RepCounter:
    """Real-time repetition counter using segmentation models."""
    def __init__(self, models_dir: str = "models/segmentation"):
        self.models_dir = models_dir
        self.models = {}
        self.extractors = {}
        self.feature_buffers = {}
        self.probability_buffers = {}
        self.last_prob = {}
        self.csv_file = open("probabilities.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "timestamp", "exercise", "probability"])
        self.frame_counter = 0
        exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips']
        for exercise in exercise_types:
            model_path = os.path.join(models_dir, f"{exercise}.keras")
            if os.path.exists(model_path):
                self.models[exercise] = tf.keras.models.load_model(model_path, compile=False)
                self.extractors[exercise] = FeatureExtractor()
                self.feature_buffers[exercise] = deque(maxlen=30)
                self.probability_buffers[exercise] = deque(maxlen=30)
                self.last_prob[exercise] = 0.0
                print(f"Loaded {exercise} segmentation model")
            else:
                print(f"Warning: {exercise} segmentation model not found")

    def update_with_window(self, window: np.ndarray, exercise_type: str) -> float:
        if exercise_type not in self.models:
            return 0.0
        sequence = window[None, ...]
        probability = self.models[exercise_type].predict(sequence, verbose=0)
        prob_value = float(probability[0, -1, 0])
        self.last_prob[exercise_type] = prob_value
        self.frame_counter += 1
        self.csv_writer.writerow([self.frame_counter, time.time(), exercise_type, prob_value])
        if self.frame_counter % 10 == 0:
            self.csv_file.flush()
        return prob_value

    def close(self):
        self.csv_file.close() 