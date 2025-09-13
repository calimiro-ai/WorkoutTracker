#!/usr/bin/env python3
"""
Multitask Trainer for 3 FPS Processing

Handles training of the multitask TCN model with 3 FPS processing.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .model import build_multitask_tcn_model
from .dataset_builder import MultitaskDatasetBuilder


class MultitaskTrainer:
    """
    Trainer for multitask TCN model with 3 FPS processing.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", 
                 target_fps: int = 3, experiment_name: str = "multitask_tcn", 
                 output_dir: str = "models"):
        """
        Initialize the multitask trainer.
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.target_fps = target_fps
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, experiment_name), exist_ok=True)
        
        # Initialize dataset builder
        self.dataset_builder = MultitaskDatasetBuilder(
            videos_dir=videos_dir,
            labels_dir=labels_dir,
            target_fps=target_fps
        )
        
        # Model configuration
        self.model_config = {
            'input_dim': 25,
            'num_classes': 5,
            'window_size': 60,
            'backbone_filters': 128,
            'backbone_layers': 6,
            'kernel_size': 3,
            'dropout_rate': 0.2,
            'classification_units': [128, 64],
            'segmentation_units': [64, 32],
            'use_attention': True,
            'classification_weight': 1.0,
            'segmentation_weight': 1.0,
        }
        
        print(f"Multitask trainer (3 FPS) initialized:")
        print(f"  Target FPS: {target_fps}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Output directory: {os.path.join(output_dir, experiment_name)}")
    
    def prepare_data(self, balance_classes: bool = True) -> Tuple[Any, Any]:
        """
        Prepare training and validation data.
        """
        print("Preparing multitask dataset with 3 FPS processing...")
        
        # Build dataset
        X, y_classification_output, y_segmentation_output, metadata = self.dataset_builder.build(balance_classes)
        
        # Update model config with actual data info
        self.model_config['num_classes'] = metadata['num_classes']
        self.model_config['input_dim'] = metadata['input_dim']
        self.model_config['window_size'] = metadata['window_size']
        
        print(f"Dataset prepared:")
        print(f"  Total samples: {len(X)}")
        print(f"  Input shape: {X.shape}")
        print(f"  Processing FPS: {metadata['processing_fps']}")
        print(f"  Frame skip: {metadata['frame_skip']}")
        
        # Split data
        X_train, X_temp, y_cls_train, y_cls_temp, y_seg_train, y_seg_temp = train_test_split(
            X, y_classification_output, y_segmentation_output, 
            test_size=0.3, random_state=42, stratify=y_classification_output
        )
        
        X_val, X_test, y_cls_val, y_cls_test, y_seg_val, y_seg_test = train_test_split(
            X_temp, y_cls_temp, y_seg_temp,
            test_size=0.5, random_state=42, stratify=y_cls_temp
        )
        
        print(f"Data splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Create datasets
        train_data = (X_train, y_cls_train, y_seg_train)
        val_data = (X_val, y_cls_val, y_seg_val)
        
        return train_data, val_data
    
    def build_model(self):
        """Build the multitask TCN model."""
        print("Building multitask TCN model...")
        return build_multitask_tcn_model(**self.model_config)
    
    def train(self, train_data: Tuple, val_data: Tuple, 
              epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 1e-3, patience: int = 20) -> tf.keras.Model:
        """
        Train the multitask model.
        """
        X_train, y_cls_train, y_seg_train = train_data
        X_val, y_cls_val, y_seg_val = val_data
        
        # Build model
        model = self.build_model()
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'classification_output': 'sparse_categorical_crossentropy',
                'segmentation_output': 'binary_crossentropy'
            },
            metrics={
                'classification_output': ['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)],
                'segmentation_output': ['accuracy']
            }
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, self.experiment_name, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        print(f"Training model for {epochs} epochs...")
        history = model.fit(
            X_train,
            {'classification_output': y_cls_train, 'segmentation_output': y_seg_train},
            validation_data=(X_val, {'classification_output': y_cls_val, 'segmentation_output': y_seg_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return model
