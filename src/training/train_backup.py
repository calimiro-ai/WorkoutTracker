#!/usr/bin/env python3
"""
Train Multitask TCN Model with 3 FPS Processing

Trains the unified TCN model for both exercise classification and segmentation
with 3 FPS processing for consistency with real-time demos.
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
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Multitask TCN with 3 FPS Processing')
    parser.add_argument('--videos-dir', default='data/raw', help='Directory containing training videos')
    parser.add_argument('--labels-dir', default='data/labels', help='Directory containing manual labels')
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument('--experiment-name', help='Name for this training experiment')
    parser.add_argument('--target-fps', type=int, default=3, help='Target processing FPS (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--balance-classes', action='store_true', help='Balance classes during training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MULTITASK TCN TRAINING - 3 FPS PROCESSING")
    print("="*60)
    print(f"Videos directory: {args.videos_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Balance classes: {args.balance_classes}")
    
    # Model configuration
    model_config = {
        'num_classes': 4,  # Will be updated based on dataset
        'window_size': 30,
        'num_features': 25,
        'backbone_filters': 64,
        'backbone_layers': 4,
        'backbone_dilation_rate': 2,
        'use_attention': True,
        'attention_heads': 4,
        'dropout_rate': 0.2,
        'l2_reg': 1e-4
    }
    
    # Training configuration
    training_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'min_delta': 1e-4,
        'validation_split': 0.2,
        'test_split': 0.1,
        'class_weight': None,
        'segmentation_weight': 1.0,
        'classification_weight': 1.0
    }
    
    # Create trainer
    trainer = MultitaskTrainer(
        model_config=model_config,
        training_config=training_config,
        videos_dir=args.videos_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        target_fps=args.target_fps,
        balance_classes=args.balance_classes
    )
    
    try:
        # Prepare data
        train_data, val_data = trainer.prepare_data(
            balance_classes=args.balance_classes
        )
        
        # Train model
        model = trainer.train(train_data, val_data)
        
        # Evaluate model
        results = trainer.evaluate(model)
        
        # Create production model
        production_path = trainer.create_production_model(model)
        
        # Save results
        results_path = os.path.join(trainer.experiment_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment directory: {trainer.experiment_dir}")
        print(f"Production model: {production_path}")
        print("Final Results:")
        print(f"  Classification Accuracy: {results['classification_accuracy']:.3f}")
        print(f"  Segmentation Accuracy: {results['segmentation_accuracy']:.3f}")
        print(f"  Segmentation AUC-ROC: {results['segmentation_auc_roc']:.3f}")
        print(f"  Segmentation AUC-PR: {results['segmentation_auc_pr']:.3f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
