"""
Multitask Training Module

Contains unified TCN models and training code for simultaneous exercise classification 
and repetition segmentation tasks.
"""

from .model import build_multitask_tcn_model
from .trainer import MultitaskTrainer
from .dataset_builder import MultitaskDatasetBuilder

__all__ = [
    'build_multitask_tcn_model',
    'MultitaskTrainer', 
    'MultitaskDatasetBuilder'
] 