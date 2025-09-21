# Data Directory

This directory contains all data files for the WorkoutTracker system.

## 📁 Structure

```
data/
├── raw/                    # Original exercise videos
│   ├── push-ups/          # Push-up training videos
│   ├── squats/            # Squat training videos
│   ├── pull-ups/          # Pull-up training videos
│   ├── dips/              # Dip training videos
│   └── no-exercise/       # Non-exercise videos
├── labels/                # Manual CSV labels
│   ├── push-ups/          # Push-up repetition labels
│   ├── squats/            # Squat repetition labels
│   ├── pull-ups/          # Pull-up repetition labels
│   ├── dips/              # Dip repetition labels
│   └── no_exercise/       # No-exercise labels
├── processed/             # Generated datasets
│   └── multitask_dataset.npz
├── test_videos/           # Test videos for demos
└── no_exercise/           # Additional no-exercise data
```

## 🎥 Raw Videos (`raw/`)

### Video Requirements
- **Format**: MP4 (H.264 codec recommended)
- **Resolution**: 640x480 or higher
- **Frame Rate**: 30 FPS
- **Duration**: 10-60 seconds per video
- **Quality**: Clear, well-lit, stable camera

### Exercise Categories

#### Push-ups (`push-ups/`)
- Standard push-up variations
- Different angles and positions
- Various rep counts (5-50+)
- Different body types and fitness levels

#### Squats (`squats/`)
- Standard squat variations
- Different depths and speeds
- Various rep counts (5-50+)
- Different body types and fitness levels

#### Pull-ups (`pull-ups/`)
- Standard pull-up variations
- Different grip positions
- Various rep counts (1-20+)
- Different body types and fitness levels

#### Dips (`dips/`)
- Standard dip variations
- Different angles and positions
- Various rep counts (1-30+)
- Different body types and fitness levels

#### No-Exercise (`no-exercise/`)
- Walking, standing, sitting
- Daily activities
- Non-exercise movements
- Background activities

## 🏷️ Labels (`labels/`)

### Label Format
Each video has a corresponding CSV file with frame-by-frame labels:

```csv
frame,label
0,0
1,0
2,1
3,0
4,0
...
```

### Label Values
- **0**: No repetition (rest, setup, transition)
- **1**: Repetition start marker

### Labeling Guidelines

#### When to Mark Repetition Start
- **Push-ups**: When chest touches ground
- **Squats**: When hips reach lowest point
- **Pull-ups**: When chin clears the bar
- **Dips**: When shoulders reach lowest point

#### Labeling Best Practices
1. **Consistency**: Use same criteria for all videos
2. **Accuracy**: Mark exact frame of repetition start
3. **Completeness**: Label all repetitions in video
4. **Quality**: Double-check labels for accuracy

## 📊 Processed Data (`processed/`)

### Multitask Dataset (`multitask_dataset.npz`)

**File Contents:**
- `X`: Feature array (N, 30, 25)
- `y_classification`: Class labels (N,)
- `y_segmentation`: Segmentation labels (N, 30)
- `metadata`: Dataset information

**Feature Details:**
- **N**: Number of sequences (~46,700)
- **30**: Frames per sequence (1 second at 30 FPS)
- **25**: Joint angles per frame

**Class Mapping:**
- 0: push-ups
- 1: squats
- 2: pull-ups
- 3: dips
- 4: no-exercise

**Segmentation Labels:**
- Gaussian-augmented probabilities (0.0-1.0)
- Center (rep start): 1.0
- ±4 frames: 0.5
- ±12 frames: ~0.1

## 🧪 Test Videos (`test_videos/`)

### Purpose
Test videos for demo applications and model evaluation.

### Test Video Types
- **test0.mp4**: Basic push-up test
- **test1.mp4**: Squat test
- **test2.mp4**: Pull-up test
- **test3.mp4**: Dip test
- **test4.mp4**: Mixed exercise test
- **test5.mp4**: Complex movement test
- **test6.mp4**: Challenging detection test

### Test Video Characteristics
- **Duration**: 10-30 seconds
- **Quality**: High resolution, good lighting
- **Content**: Clear exercise demonstrations
- **Variety**: Different angles, speeds, forms

## 📈 Dataset Statistics

### Current Dataset Size
- **Total Videos**: ~200+ exercise videos
- **Total Sequences**: 46,700
- **Total Frames**: 1,401,000
- **Total Repetitions**: ~16,640

### Class Distribution
- **Push-ups**: 13,630 sequences (29.2%)
- **Squats**: 14,666 sequences (31.4%)
- **Pull-ups**: 8,221 sequences (17.6%)
- **Dips**: 10,183 sequences (21.8%)
- **No-exercise**: 0 sequences (filtered during training)

### Temporal Distribution
- **Window Size**: 30 frames (1 second)
- **Stride**: 1 frame (overlapping windows)
- **Positive Samples**: 9.9% of all frames
- **Augmentation**: Gaussian smoothing applied

## 🔧 Data Management

### Adding New Videos
1. Place videos in appropriate `raw/` subdirectory
2. Create corresponding labels in `labels/` subdirectory
3. Run `python build_dataset.py` to update dataset
4. Retrain model with `python train.py`

### Data Validation
```bash
# Check video properties
python -c "
import cv2
cap = cv2.VideoCapture('data/raw/push-ups/video.mp4')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
print(f'Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
print(f'Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
"
```

### Label Validation
```bash
# Check label consistency
python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('data/labels/push-ups/video.csv')
print(f'Total frames: {len(df)}')
print(f'Repetitions: {df[\"label\"].sum()}')
print(f'Label range: {df[\"label\"].min()}-{df[\"label\"].max()}')
"
```

## 🐛 Troubleshooting

### Common Issues

1. **Video Format**: Convert to MP4 if needed
2. **Label Mismatch**: Ensure CSV matches video frame count
3. **Missing Data**: Check file paths and permissions
4. **Corrupted Files**: Re-download or re-process videos

### Data Quality Checks

**Video Quality:**
- Check resolution and frame rate
- Verify audio/video sync
- Test playback compatibility
- Validate file integrity

**Label Quality:**
- Verify frame count alignment
- Check label value ranges
- Validate temporal consistency
- Review manual accuracy

## 📚 Data Sources

### Original Data
- **Exercise Videos**: Self-recorded and collected
- **Labeling**: Manual frame-by-frame annotation
- **Quality Control**: Multiple review passes
- **Validation**: Cross-checking and verification

### Data Augmentation
- **Gaussian Smoothing**: Temporal label augmentation
- **Balanced Sampling**: Class distribution balancing
- **Window Sliding**: Overlapping sequence generation
- **Noise Handling**: Robust pose detection

## 🔒 Data Privacy

### Privacy Considerations
- **Personal Videos**: Anonymized and de-identified
- **Consent**: All participants provided consent
- **Usage**: Research and educational purposes only
- **Storage**: Secure local storage only

### Data Sharing
- **Public Release**: Not currently planned
- **Research Use**: Available for academic research
- **Collaboration**: Contact project maintainers
- **Licensing**: Subject to project license terms
