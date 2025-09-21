"""
SmartMirror - Real-time Exercise Detection and Rep Counting

A computer vision system for real-time exercise classification and repetition counting
using MediaPipe pose detection and deep learning models.
"""

__version__ = "1.0.0"
__author__ = "SmartMirror Team"

from .src.core.realtime_pipeline import RealTimePipeline
from .src.core.dataset_builder import MultitaskDatasetBuilder

__all__ = [
    'RealTimePipeline',
    'MultitaskDatasetBuilder',
] 
