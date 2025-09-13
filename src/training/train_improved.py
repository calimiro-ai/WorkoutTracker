#!/usr/bin/env python3
"""
Train Multitask TCN Model with Improved Configuration

Improved model architecture and training parameters for better performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.trainer import MultitaskTrainer

def main():
    """Main training function with improved configuration."""
    parser = argparse.ArgumentParser(description='Train Multitask TCN with Improved Configuration')
    parser.add_argument('--videos-dir', default='data/raw', help='Directory containing training videos')
    parser.add_argument('--labels-dir', default='data/labels', help='Directory containing manual labels')
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument('--experiment-name', help='Name for this training experiment')
    parser.add_argument('--target-fps', type=int, default=3, help='Target processing FPS (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--balance-classes', action='store_true', help='Balance classes during training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MULTITASK TCN TRAINING - IMPROVED CONFIGURATION")
    print("="*60)
    print(f"Videos directory: {args.videos_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patience: {args.patience}")
    print(f"Balance classes: {args.balance_classes}")
    
    # Create trainer with improved configuration
    trainer = MultitaskTrainer(
        videos_dir=args.videos_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        target_fps=args.target_fps
    )
    
    try:
        # Prepare data
        train_data, val_data = trainer.prepare_data(
            balance_classes=args.balance_classes
        )
        
        # Train model with improved parameters
        model = trainer.train(
            train_data, 
            val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment directory: {trainer.experiment_dir}")
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
