#!/usr/bin/env python3
"""
Train improved recall model with no_exercise data included.
"""

import sys
import os
sys.path.append('src')

from training.trainer import MultitaskTrainer

def main():
    print("Starting improved recall training...")
    
    # Create trainer
    trainer = MultitaskTrainer(
        experiment_name="main"
    )
    
    # Run training
    model = trainer.run_training(
        epochs=250,
        batch_size=32,
        use_balanced_sampling=True,
        positive_ratio=0.2,
        patience=20,
        window_size=30
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
