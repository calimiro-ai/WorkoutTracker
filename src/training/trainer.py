#!/usr/bin/env python3
"""
Multitask Trainer for 30 FPS Processing

Handles training of the multitask TCN model with 30 FPS processing.
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
    Trainer for multitask TCN model with 30 FPS processing.
    """
    
    def __init__(self, videos_dir: str = "data/raw", labels_dir: str = "data/labels", 
                 fps: int = 30, experiment_name: str = "multitask_tcn_30fps", 
                 output_dir: str = "models"):
        """
        Initialize the multitask trainer.
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
        self.dataset_path = "data/processed/multitask_dataset_30fps.npz"
        
        # Model configuration
        self.model_config = {
            'input_dim': 25,
            'num_classes': 5,
            'window_size': 30,  # 30 frames = 1 second at 30fps
            'backbone_filters': 128,
            'backbone_layers': 6,
            'kernel_size': 3,
            'dropout_rate': 0.2,
            'classification_units': [64, 32],
            'segmentation_units': [64, 32],
            'classification_weight': 1.0,
            'segmentation_weight': 1.0,
        }
        
        print(f"Multitask trainer initialized:")
        print(f"  FPS: {fps}")
        print(f"  Window size: {self.model_config['window_size']} frames")
        print(f"  Experiment: {experiment_name}")
        print(f"  Output dir: {self.experiment_dir}")
        print(f"  Dataset: {self.dataset_path}")
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load the multitask dataset.
        
        Returns:
            Tuple of (X, y_classification, y_segmentation, metadata)
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"Loading dataset from {self.dataset_path}...")
        data = np.load(self.dataset_path, allow_pickle=True)
        
        X = data['X']
        y_classification = data['y_classification']
        y_segmentation = data['y_segmentation']
        metadata = data['metadata'].item()
        
        print(f"Dataset loaded:")
        print(f"  Features shape: {X.shape}")
        print(f"  Classification labels shape: {y_classification.shape}")
        print(f"  Segmentation labels shape: {y_segmentation.shape}")
        print(f"  Exercise types: {metadata['exercise_types']}")
        print(f"  Processing FPS: {metadata.get('fps', 'unknown')}")
        
        return X, y_classification, y_segmentation, metadata
    
    def prepare_data(self, X: np.ndarray, y_classification: np.ndarray, 
                    y_segmentation: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting into train/validation sets.
        
        Args:
            X: Feature sequences
            y_classification: Classification labels
            y_segmentation: Segmentation labels
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, y_cls_train, y_cls_val, y_seg_train, y_seg_val)
        """
        print(f"Preparing data with {test_size:.1%} validation split...")
        
        # Split the data
        X_train, X_val, y_cls_train, y_cls_val = train_test_split(
            X, y_classification, test_size=test_size, random_state=random_state, stratify=y_classification
        )
        
        # Split segmentation labels accordingly
        _, _, y_seg_train, y_seg_val = train_test_split(
            X, y_segmentation, test_size=test_size, random_state=random_state, stratify=y_classification
        )
        
        print(f"Data split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_cls_train, y_cls_val, y_seg_train, y_seg_val
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the multitask TCN model.
        
        Returns:
            Compiled Keras model
        """
        print("Creating multitask TCN model...")
        
        model = build_multitask_tcn_model(**self.model_config)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification_output': SparseCategoricalCrossentropy(from_logits=False),
                'segmentation_output': BinaryFocalCrossentropy(alpha=0.99, gamma=5.0)
            },
            loss_weights={
                'classification_output': self.model_config['classification_weight'],
                'segmentation_output': self.model_config['segmentation_weight']
            },
            metrics={
                'classification_output': ['accuracy'],
                'segmentation_output': ['binary_accuracy']
            }
        )
        
        print(f"Model created with {model.count_params():,} parameters")
        return model
    
    def train(self, X_train: np.ndarray, y_cls_train: np.ndarray, y_seg_train: np.ndarray,
              X_val: np.ndarray, y_cls_val: np.ndarray, y_seg_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32, use_balanced_sampling: bool = True,
              positive_ratio: float = 0.3) -> tf.keras.Model:
        """
        Train the multitask model.
        
        Args:
            X_train: Training features
            y_cls_train: Training classification labels
            y_seg_train: Training segmentation labels
            X_val: Validation features
            y_cls_val: Validation classification labels
            y_seg_val: Validation segmentation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_balanced_sampling: Whether to use balanced sampling for segmentation
            positive_ratio: Ratio of positive sequences in each batch
            
        Returns:
            Trained model
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Use balanced sampling: {use_balanced_sampling}")
        if use_balanced_sampling:
            print(f"Positive ratio: {positive_ratio:.1%}")
        
        # Create model
        model = self.create_model()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.experiment_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Train the model
        if use_balanced_sampling:
            print(f"Using balanced sampling with {positive_ratio:.1%} positive ratio per batch")
            
            # Create balanced training generator
            from .balanced_generator import BalancedSequenceGenerator
            train_gen = BalancedSequenceGenerator(
                X_train, y_cls_train, y_seg_train,
                batch_size=batch_size,
                positive_ratio=positive_ratio,
                shuffle=True
            )
            
            # Use validation data directly (not balanced for more accurate validation)
            val_data_dict = {'classification_output': y_cls_val, 'segmentation_output': y_seg_val}
            
            history = model.fit(
                train_gen,
                validation_data=(X_val, val_data_dict),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("Using standard training (no balanced sampling)")
            history = model.fit(
                X_train,
                {'classification_output': y_cls_train, 'segmentation_output': y_seg_train},
                validation_data=(X_val, {'classification_output': y_cls_val, 'segmentation_output': y_seg_val}),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save training history
        np.save(os.path.join(self.experiment_dir, 'training_history.npy'), history.history)
        
        print(f"Training completed. Model saved to {self.experiment_dir}")
        return model
    
    def run_full_training(self, epochs: int = 100, batch_size: int = 32, 
                         use_balanced_sampling: bool = True, positive_ratio: float = 0.3) -> tf.keras.Model:
        """
        Run the complete training pipeline.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_balanced_sampling: Whether to use balanced sampling
            positive_ratio: Ratio of positive sequences in each batch
            
        Returns:
            Trained model
        """
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
            epochs=epochs, batch_size=batch_size,
            use_balanced_sampling=use_balanced_sampling,
            positive_ratio=positive_ratio
        )
        
        return model


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multitask TCN model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--no_balanced_sampling', action='store_true', 
                       help='Disable balanced sampling')
    parser.add_argument('--positive_ratio', type=float, default=0.3,
                       help='Ratio of positive sequences in each batch')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultitaskTrainer()
    
    # Run training
    model = trainer.run_full_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_balanced_sampling=not args.no_balanced_sampling,
        positive_ratio=args.positive_ratio
    )
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
