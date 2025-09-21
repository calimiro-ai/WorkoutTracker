#!/usr/bin/env python3
"""
Multitask Trainer

Includes no_exercise data with rep probability 0, better class balancing,
and more negative samples for improved recall.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryFocalCrossentropy
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .model import build_multitask_tcn_model
from .balanced_generator import create_balanced_generators
import numpy as np


class MultitaskTrainer:
    """
    Improved trainer for multitask TCN model with better recall.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", 
                 fps: int = 30, experiment_name: str = "main", 
                 output_dir: str = "models"):
        """
        Initialize the improved multitask trainer.
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.fps = fps
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        
        # Create output directory
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Dataset file path
        self.dataset_path = "data/processed/multitask_dataset.npz"
        
        # Improved model configuration for better recall
        self.model_config = {
            'input_dim': 25,
            'num_classes': 5,  # Include no_exercise class
            'window_size': 30,
            'backbone_filters': 128,
            'backbone_layers': 8,
            'kernel_size': 3,
            'dropout_rate': 0.25,
            'classification_units': [128, 64, 32],
            'segmentation_units': [128, 64, 32],
            'use_attention': True,
            'classification_weight': 1.0,
            'segmentation_weight': 5.0,
            'learning_rate': 5e-4,
        }
        
        # Balanced focal loss configuration
        self.focal_loss_config = {
            'gamma': 1.0,
            'alpha': 0.5  # More balanced between classes
        }
        
        print(f"Improved multitask trainer initialized:")
        print(f"  FPS: {fps}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Output directory: {self.experiment_dir}")
        print(f"  Classes: {self.model_config['num_classes']} (including no_exercise)")
        print(f"  Focal loss: gamma={self.focal_loss_config['gamma']}, alpha={self.focal_loss_config['alpha']}")
        print(f"  Segmentation weight: {self.model_config['segmentation_weight']}")
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load the multitask dataset.
        
        Returns:
            Tuple of (features, classification_labels, segmentation_labels, metadata)
        """
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found at {self.dataset_path}")
            print("Please run the dataset builder first:")
            print("python src/core/dataset_builder.py")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"Loading dataset from {self.dataset_path}...")
        
        # Load dataset
        data = np.load(self.dataset_path, allow_pickle=True)
        X = data['X']
        y_classification = data['y_classification']
        y_segmentation = data['y_segmentation']
        metadata = data['metadata'].item()
        
        print(f"Dataset loaded successfully:")
        print(f"  Features shape: {X.shape}")
        print(f"  Classification labels shape: {y_classification.shape}")
        print(f"  Segmentation labels shape: {y_segmentation.shape}")
        
        # Print class distribution
        unique_classes, class_counts = np.unique(y_classification, return_counts=True)
        print(f"  Class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = count / len(y_classification) * 100
            print(f"    Class {cls}: {count} samples ({percentage:.1f}%)")
        
        # Print segmentation statistics
        print(f"  Segmentation statistics:")
        print(f"    Max: {np.max(y_segmentation):.3f}")
        print(f"    Mean: {np.mean(y_segmentation):.3f}")
        print(f"    Std: {np.std(y_segmentation):.3f}")
        print(f"    Values > 0.5: {np.sum(y_segmentation > 0.5)} / {y_segmentation.size} ({np.sum(y_segmentation > 0.5)/y_segmentation.size*100:.1f}%)")
        print(f"    Values = 1.0: {np.sum(y_segmentation == 1.0)} / {y_segmentation.size} ({np.sum(y_segmentation == 1.0)/y_segmentation.size*100:.1f}%)")
        
        return X, y_classification, y_segmentation, metadata
    
    def prepare_data(self, X: np.ndarray, y_classification: np.ndarray, 
                    y_segmentation: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with train/validation split.
        """
        print("Preparing data for training...")
        
        # Split data
        X_train, X_val, y_cls_train, y_cls_val, y_seg_train, y_seg_val = train_test_split(
            X, y_classification, y_segmentation,
            test_size=test_size,
            random_state=random_state,
            stratify=y_classification
        )
        
        print(f"Data split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_cls_train, y_cls_val, y_seg_train, y_seg_val
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the multitask TCN model.
        """
        print("Creating multitask TCN model...")
        
        model = build_multitask_tcn_model(**self.model_config)
        
        # Compile model with improved loss functions
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
            loss={
                'classification_output': SparseCategoricalCrossentropy(),
                'segmentation_output': BinaryFocalCrossentropy(
                    gamma=self.focal_loss_config['gamma'],
                    alpha=self.focal_loss_config['alpha']
                )
            },
            loss_weights={
                'classification_output': self.model_config['classification_weight'],
                'segmentation_output': self.model_config['segmentation_weight']
            },
            metrics={
                'classification_output': ['accuracy'],
                'segmentation_output': ['accuracy', 'precision', 'recall', 'auc']
            }
        )
        
        print("Model created and compiled successfully!")
        return model
    
    def train(self, X_train: np.ndarray, y_cls_train: np.ndarray, y_seg_train: np.ndarray,
              X_val: np.ndarray, y_cls_val: np.ndarray, y_seg_val: np.ndarray,
              epochs: int = 250, batch_size: int = 32, use_balanced_sampling: bool = True,
              positive_ratio: float = 0.2, patience: int = 20, window_size: int = 30) -> tf.keras.Model:
        """
        Train the improved multitask model with better recall.
        """
        print("Starting improved training for better recall...")
        print(f"Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Patience: {patience}")
        print(f"  Balanced sampling: {use_balanced_sampling}")
        print(f"  Positive ratio: {positive_ratio}")
        
        # Update model config with provided window_size
        self.model_config['window_size'] = window_size
        print(f"  Using window_size: {window_size}")
        
        # Create model
        model = self.create_model()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_segmentation_output_loss', mode='min',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_segmentation_output_loss', mode='min',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.experiment_dir, 'best_model.keras'),
                monitor='val_segmentation_output_loss', mode='min',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare data generators
        if use_balanced_sampling:
            print("Using balanced sampling for training...")
            train_gen, val_gen = create_balanced_generators(
                X_train, y_cls_train, y_seg_train,
                batch_size=batch_size,
                positive_ratio=positive_ratio
            )
            
            # Train with balanced generators
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without balanced sampling
            history = model.fit(
                X_train, {'classification_output': y_cls_train, 'segmentation_output': y_seg_train},
                validation_data=(X_val, {'classification_output': y_cls_val, 'segmentation_output': y_seg_val}),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save training history
        history_path = os.path.join(self.experiment_dir, 'training_history.npy')
        np.save(history_path, history.history)
        print(f"Training history saved to: {history_path}")
        
        print("Training completed!")
        return model
    
    def run_training(self, epochs: int = 250, batch_size: int = 32, 
                    use_balanced_sampling: bool = True, positive_ratio: float = 0.2,
                    patience: int = 20, window_size: int = 30) -> tf.keras.Model:
        """
        Run the complete training pipeline.
        """
        print("="*60)
        print("IMPROVED MULTITASK TRAINER - BETTER RECALL")
        print("="*60)
        
        # Load dataset
        X, y_classification, y_segmentation, metadata = self.load_dataset()
        
        # Prepare data
        X_train, X_val, y_cls_train, y_cls_val, y_seg_train, y_seg_val = self.prepare_data(
            X, y_classification, y_segmentation
        )
        
        # Train model
        model = self.train(
            X_train, y_cls_train, y_seg_train,
            X_val, y_cls_val, y_seg_val,
            epochs=epochs,
            batch_size=batch_size,
            use_balanced_sampling=use_balanced_sampling,
            positive_ratio=positive_ratio,
            patience=patience,
            window_size=window_size
        )
        
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Experiment directory: {self.experiment_dir}")
        print("Model saved successfully!")
        
        return model


if __name__ == "__main__":
    # Create trainer
    trainer = MultitaskTrainer(
        experiment_name="main"
    )
    
    # Run training
    model = trainer.run_training(
        epochs=250,
        batch_size=32,
        use_balanced_sampling=True,
        positive_ratio=0.2,  # More negative samples for better recall
        patience=20,
        window_size=30
    )
