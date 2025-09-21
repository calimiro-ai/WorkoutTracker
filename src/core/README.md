# Dataset Building

This directory contains the core data processing and dataset building functionality.

## �� Contents

### Core Files

- **`dataset_builder.py`**: Main dataset builder with Gaussian augmentation
- **`improved_dataset_builder.py`**: Alternative implementation
- **`build_dataset.py`**: Script to build the multitask dataset

## 🏗️ Dataset Pipeline

### 1. Feature Extraction

**MediaPipe Pose Detection:**
- Extracts 25 joint angles from each video frame
- Handles pose tracking and landmark detection
- Processes videos at 30 FPS

**Joint Angle Calculation:**
- Computes relative angles between connected joints
- Normalizes angles for consistent representation
- Handles missing poses gracefully

### 2. Temporal Windowing

**Sliding Window Approach:**
- Window size: 30 frames (1 second at 30 FPS)
- Stride: 1 frame (overlapping windows)
- Creates sequences for temporal analysis

**Window Statistics:**
- Total sequences: ~46,700
- Features per sequence: 30 × 25 = 750
- Memory efficient processing

### 3. Label Processing

**Manual Labeling:**
- CSV files with frame-by-frame labels
- Binary markers for repetition starts
- Metadata for video information

**Gaussian Augmentation:**
- Smooths binary labels into probability distributions
- Center (rep start): 1.0 probability
- ±4 frames: 0.5 probability
- ±12 frames: ~0.1 probability
- Creates realistic temporal patterns

## 🔧 Configuration

### Dataset Builder Parameters

```python
MultitaskDatasetBuilder(
    videos_dir="data/raw",           # Input video directory
    no_exercise_dir="data/no_exercise",  # No-exercise videos
    labels_dir="data/labels",        # Label CSV files
    fps=30,                          # Video frame rate
    window_size=30,                  # Sequence length
    margin_frames=12,                # Gaussian augmentation margin
    sigma=3.40,                      # Gaussian standard deviation
    no_exercise_ratio=0.3           # No-exercise sample ratio
)
```

### Label Augmentation

**Gaussian Parameters:**
- `margin_frames`: ±12 frames around rep markers
- `sigma`: 3.40 for smooth probability decay
- Creates natural temporal transitions

**Augmentation Formula:**
```
probability = exp(-0.5 * (frame_distance / sigma)²)
```

## 📊 Dataset Structure

### Input Data

**Video Organization:**
```
data/raw/
├── push-ups/          # Push-up videos
├── squats/            # Squat videos
├── pull-ups/          # Pull-up videos
├── dips/              # Dip videos
└── no-exercise/       # Non-exercise videos
```

**Label Organization:**
```
data/labels/
├── push-ups/          # Push-up labels (CSV)
├── squats/            # Squat labels (CSV)
├── pull-ups/          # Pull-up labels (CSV)
├── dips/              # Dip labels (CSV)
└── no_exercise/       # No-exercise labels (CSV)
```

### Output Dataset

**File: `data/processed/multitask_dataset.npz`**

**Contents:**
- `X`: Features array (N, 30, 25)
- `y_classification`: Class labels (N,)
- `y_segmentation`: Segmentation labels (N, 30)
- `metadata`: Dataset information

**Class Mapping:**
- 0: push-ups
- 1: squats
- 2: pull-ups
- 3: dips
- 4: no-exercise

## 🚀 Usage

### Building Dataset

```bash
# Basic usage
python build_dataset.py

# With custom parameters
python -c "
from src.core.dataset_builder import MultitaskDatasetBuilder
builder = MultitaskDatasetBuilder(
    window_size=60,      # Longer sequences
    margin_frames=20,    # Larger augmentation
    sigma=5.0            # Smoother curves
)
X, y_cls, y_seg, meta = builder.build_dataset()
"
```

### Custom Configuration

**Modify `build_dataset.py`:**
```python
builder = MultitaskDatasetBuilder(
    videos_dir="custom/raw",
    labels_dir="custom/labels",
    window_size=45,          # 1.5 second windows
    margin_frames=15,        # Larger augmentation
    sigma=4.0,               # Different smoothness
    no_exercise_ratio=0.4    # More no-exercise samples
)
```

## 📈 Dataset Statistics

### Current Dataset

**Size:**
- Total sequences: 46,700
- Training samples: 37,360 (80%)
- Validation samples: 9,340 (20%)

**Class Distribution:**
- Push-ups: 13,630 (29.2%)
- Squats: 14,666 (31.4%)
- Pull-ups: 8,221 (17.6%)
- Dips: 10,183 (21.8%)
- No-exercise: 0 (filtered during training)

**Segmentation Statistics:**
- Positive samples: 9.9% of all frames
- Peak probability: 1.0 (rep starts)
- Mean probability: 0.092
- Standard deviation: 0.242

## 🔧 Advanced Features

### Custom Augmentation

**Modify `LabelAugmenter`:**
```python
class CustomLabelAugmenter(LabelAugmenter):
    def augment(self, labels):
        # Custom augmentation logic
        # e.g., different probability curves
        pass
```

### Multi-Resolution

**Different Window Sizes:**
```python
# Short sequences (1 second)
builder_short = MultitaskDatasetBuilder(window_size=30)

# Long sequences (2 seconds)  
builder_long = MultitaskDatasetBuilder(window_size=60)
```

### Data Validation

**Quality Checks:**
- Verify video-label alignment
- Check for missing poses
- Validate label consistency
- Monitor augmentation quality

## 🐛 Troubleshooting

### Common Issues

1. **Missing Labels**: Ensure CSV files exist for all videos
2. **Pose Detection Failures**: Check video quality and lighting
3. **Memory Issues**: Process videos in batches
4. **Label Mismatch**: Verify video-label correspondence

### Debug Mode

Enable verbose processing:
```python
builder = MultitaskDatasetBuilder(verbose=True)
```

### Data Inspection

Check dataset contents:
```python
import numpy as np
data = np.load('data/processed/multitask_dataset.npz')
print(f"Features shape: {data['X'].shape}")
print(f"Classes: {np.unique(data['y_classification'])}")
print(f"Segmentation range: {data['y_segmentation'].min():.3f} - {data['y_segmentation'].max():.3f}")
```
