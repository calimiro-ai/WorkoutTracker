#!/usr/bin/env python3
"""
Improved Signal Processing Module for Exercise Repetition Detection using scipy.signal
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, List, Dict


class SignalProcessor:
    """Improved signal processor using scipy.signal for robust peak detection."""
    
    def __init__(self, rep_threshold: float = 0.5, min_peak_distance: int = 5):
        """Initialize the signal processor."""
        self.rep_threshold = rep_threshold
        self.min_peak_distance = min_peak_distance
        
        # History storage for each exercise
        self.probability_history: Dict[str, List[float]] = {}
        self.frame_history: Dict[str, List[int]] = {}
        self.last_rep_frame: Dict[str, int] = {}
        self.rep_counts: Dict[str, int] = {}
        self.detected_peaks: Dict[str, List[int]] = {}
    
    def initialize_exercise(self, exercise: str):
        """Initialize tracking for a new exercise type."""
        if exercise not in self.probability_history:
            self.probability_history[exercise] = []
            self.frame_history[exercise] = []
            self.last_rep_frame[exercise] = -1
            self.rep_counts[exercise] = 0
            self.detected_peaks[exercise] = []
    
    def detect_rep(self, exercise: str, rep_probability: float, frame_idx: int) -> bool:
        """Detect repetition using improved scipy.signal.find_peaks."""
        if exercise == "no-exercise":
            return False
        
        self.initialize_exercise(exercise)
        
        # Add current data to history
        self.probability_history[exercise].append(rep_probability)
        self.frame_history[exercise].append(frame_idx)
        
        # Need at least 15 data points for reliable peak detection
        if len(self.probability_history[exercise]) < 15:
            return False
        
        # Use scipy.signal.find_peaks for robust peak detection
        probs = np.array(self.probability_history[exercise])
        frames = np.array(self.frame_history[exercise])
        
        # Apply light smoothing to reduce noise
        if len(probs) > 5:
            probs_smooth = savgol_filter(probs, min(5, len(probs)//2*2+1), 2)
        else:
            probs_smooth = probs
        
        # Calculate relative threshold based on signal range
        signal_range = np.max(probs_smooth) - np.min(probs_smooth)
        relative_threshold = max(self.rep_threshold, np.min(probs_smooth) + 0.3 * signal_range)
        
        # Find peaks using scipy with improved parameters
        peaks, properties = find_peaks(
            probs_smooth,
            height=relative_threshold,  # Use relative threshold
            distance=self.min_peak_distance,
            prominence=0.02,  # Lower prominence for more sensitivity
            width=2,  # Minimum peak width
            rel_height=0.8  # Relative height for width calculation
        )
        
        # Check if we have new peaks that haven't been counted yet
        new_peaks = []
        for peak_idx in peaks:
            frame_at_peak = frames[peak_idx]
            if frame_at_peak not in self.detected_peaks[exercise]:
                new_peaks.append((frame_at_peak, probs_smooth[peak_idx]))
                self.detected_peaks[exercise].append(frame_at_peak)
        
        # Count new peaks as reps
        for frame_at_peak, peak_value in new_peaks:
            if frame_at_peak > self.last_rep_frame[exercise]:
                self.rep_counts[exercise] += 1
                self.last_rep_frame[exercise] = frame_at_peak
                
                print(f"REP DETECTED! {exercise}: {self.rep_counts[exercise]} "
                      f"(peak: {peak_value:.3f} at frame {frame_at_peak})")
                return True
        
        return False
    
    def get_rep_count(self, exercise: str) -> int:
        """Get current rep count for an exercise."""
        return self.rep_counts.get(exercise, 0)
    
    def get_all_peaks(self, exercise: str) -> List[Tuple[int, float]]:
        """Get all detected peaks for an exercise."""
        if exercise not in self.detected_peaks:
            return []
        
        peaks = []
        for frame in self.detected_peaks[exercise]:
            if exercise in self.frame_history:
                frame_idx = self.frame_history[exercise].index(frame)
                prob = self.probability_history[exercise][frame_idx]
                peaks.append((frame, prob))
        return peaks
    
    def reset_counts(self):
        """Reset all rep counts."""
        for exercise in self.rep_counts:
            self.rep_counts[exercise] = 0
            self.last_rep_frame[exercise] = -1
            self.detected_peaks[exercise] = []
    
    def reset_exercise(self, exercise: str):
        """Reset counts for a specific exercise."""
        if exercise in self.rep_counts:
            self.rep_counts[exercise] = 0
            self.last_rep_frame[exercise] = -1
            self.probability_history[exercise] = []
            self.frame_history[exercise] = []
            self.detected_peaks[exercise] = []
