# Model Training

This directory contains the training infrastructure for the WorkoutTracker system.

## �� Contents

### Core Files

- **`trainer.py`**: Main training class with improved recall
- **`model.py`**: TCN model architecture definition
- **`balanced_generator.py`**: Balanced data sampling for training
- **`train.py`**: Training script entry point

## 🧠 Model Architecture

### Multitask TCN Model

The model performs two tasks simultaneously:
1. **Exercise Classification**: Identify exercise type (5 classes)
2. **Repetition Segmentation**: Detect repetition frames (binary)

### Architecture Details

```
Input: (batch_size, 30, 25)
├── TCN Backbone (8 residual blocks)
│   ├── Dilated Convolutions (kernel=3)
│   ├── Residual Connections
│   ├── Layer Normalization
│   └── Dropout (0.25)
├── Multi-Head Attention (4 heads)
├── Classification Head
│   ├── Global Average Pooling
│   ├── Dense Layers [128, 64, 32]
│   └── Softmax Output (5 classes)
└── Segmentation Head
    ├── Dense Layers [128, 64, 32]
    └── Sigmoid Output (30 frames)
```

## 🚀 Training Process

### 1. Data Preparation

```bash
# Build dataset first
python build_dataset.py

# Start training
python train.py
```

### 2. Training Configuration

**Model Parameters:**
- Window size: 30 frames (1 second)
- Input features: 25 joint angles
- Classes: 5 (push-ups, squats, pull-ups, dips, no-exercise)
- Backbone filters: 128
- Backbone layers: 8
- Dropout rate: 0.25

**Training Parameters:**
- Epochs: 250
- Batch size: 32
- Learning rate: 5e-4
- Patience: 20 (early stopping)
- Positive ratio: 0.2 (balanced sampling)

**Loss Functions:**
- Classification: Focal Loss (γ=1.0, α=0.5)
- Segmentation: Binary Crossentropy
- Total: Weighted combination (seg_weight=5.0)

### 3. Training Output

**Model Files:**
- `models/main/main.keras`: Best model weights
- `models/main/training_history.npy`: Training metrics

**Metrics Tracked:**
- Classification accuracy/loss
- Segmentation accuracy/precision/recall/AUC
- Validation metrics
- Learning rate schedule

## 📊 Training Strategy

### Balanced Sampling

The `BalancedGenerator` ensures proper class distribution:
- 20% positive samples (repetition frames)
- 80% negative samples (non-repetition frames)
- Prevents model bias toward majority class

### Label Augmentation

Gaussian augmentation creates smooth temporal patterns:
- Center (rep start): Probability = 1.0
- ±4 frames: Probability = 0.5
- ±12 frames: Probability ≈ 0.1
- Improves temporal consistency

### Regularization

- **Dropout**: 0.25 rate to prevent overfitting
- **Early Stopping**: Stop when validation loss plateaus
- **Learning Rate Reduction**: Reduce LR when validation loss stops improving
- **Focal Loss**: Focus on hard examples

## 🔧 Customization

### Model Architecture

Edit `model.py` to modify:
- Number of TCN layers
- Filter sizes
- Attention heads
- Dense layer sizes

### Training Parameters

Edit `trainer.py` to modify:
- Learning rate and optimizer
- Loss function weights
- Batch size and epochs
- Callback configurations

### Data Augmentation

Edit `dataset_builder.py` to modify:
- Gaussian augmentation parameters
- Window size and stride
- Class balancing ratios

## �� Monitoring Training

### Real-time Metrics

Training displays:
- Epoch progress and timing
- Loss and accuracy metrics
- Validation performance
- Learning rate changes

### Model Checkpointing

- Best model saved automatically
- Training history preserved
- Easy model restoration

### Early Stopping

- Monitors validation segmentation loss
- Stops training when no improvement
- Prevents overfitting

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size
2. **Slow Training**: Check GPU availability
3. **Poor Convergence**: Adjust learning rate
4. **Overfitting**: Increase dropout or regularization

### Debug Mode

Enable detailed logging:
```bash
export TF_CPP_MIN_LOG_LEVEL=0
python train.py
```

### Model Evaluation

Check training progress:
```python
import numpy as np
history = np.load('models/main/training_history.npy', allow_pickle=True).item()
print(history.keys())
```

## 📚 Advanced Usage

### Custom Datasets

To train on custom data:
1. Add videos to `data/raw/`
2. Create labels in `data/labels/`
3. Run `build_dataset.py`
4. Start training with `train.py`

### Transfer Learning

To fine-tune pre-trained model:
1. Load existing weights
2. Freeze early layers
3. Train only final layers
4. Gradually unfreeze more layers

### Hyperparameter Tuning

Use tools like Optuna or Ray Tune:
1. Define parameter search space
2. Run multiple training experiments
3. Select best configuration
4. Train final model
