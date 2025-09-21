# WorkoutTracker - AI-Powered Exercise Recognition

An intelligent workout tracking system that uses computer vision and machine learning to automatically detect and count exercise repetitions in real-time videos.

## �� Overview

WorkoutTracker combines MediaPipe pose detection with a Temporal Convolutional Network (TCN) to perform two main tasks:
- **Exercise Classification**: Identify the type of exercise (push-ups, squats, pull-ups, dips, no-exercise)
- **Repetition Segmentation**: Detect and count individual repetitions within the exercise

## 🏗️ Architecture

### Data Pipeline
1. **Raw Videos** → MediaPipe pose extraction → Joint angle features
2. **Manual Labels** → CSV files with repetition start markers
3. **Gaussian Augmentation** → Smooth temporal labels around rep markers
4. **Multitask Dataset** → Combined features + classification + segmentation labels

### Model Architecture
- **TCN Backbone**: 8-layer temporal convolutional network with residual connections
- **Multi-Head Attention**: Captures global temporal dependencies
- **Dual Outputs**: Classification head + segmentation head
- **Balanced Training**: Focal loss with class balancing for better recall

## 📁 Project Structure

```
WorkoutTracker/
├── data/
│   ├── raw/                    # Original exercise videos
│   │   ├── push-ups/          # Push-up videos
│   │   ├── squats/            # Squat videos
│   │   ├── pull-ups/          # Pull-up videos
│   │   ├── dips/              # Dip videos
│   │   └── no-exercise/       # Non-exercise videos
│   ├── labels/                # Manual CSV labels
│   │   ├── push-ups/          # Push-up labels
│   │   ├── squats/            # Squat labels
│   │   ├── pull-ups/          # Pull-up labels
│   │   ├── dips/              # Dip labels
│   │   └── no_exercise/       # No-exercise labels
│   └── processed/             # Generated datasets
│       └── multitask_dataset.npz
├── models/                    # Trained models
│   └── main/                  # Current best model
│       ├── main.keras         # Model weights
│       └── training_history.npy
├── src/
│   ├── core/                  # Dataset building
│   │   ├── dataset_builder.py
│   │   └── improved_dataset_builder.py
│   ├── training/              # Model training
│   │   ├── trainer.py
│   │   ├── model.py
│   │   └── balanced_generator.py
│   ├── demo/                  # Demo applications
│   │   ├── demo.py
│   │   └── live/
│   └── utils/                 # Utilities
│       ├── video_labeler.py
│       └── csv_format_converter.py
├── demo_output/               # Demo results
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd WorkoutTracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your exercise videos in the appropriate directories:
- `data/raw/push-ups/` - Push-up videos
- `data/raw/squats/` - Squat videos  
- `data/raw/pull-ups/` - Pull-up videos
- `data/raw/dips/` - Dip videos
- `data/raw/no-exercise/` - Non-exercise videos

### 3. Label Your Data

Use the video labeling tool to create CSV files with repetition markers:

```bash
python src/utils/video_labeler.py data/raw/push-ups/video1.mp4
```

This creates `data/labels/push-ups/video1.csv` with frame-by-frame labels.

### 4. Build Dataset

Create the dataset from your labeled videos:

```bash
python build_dataset.py
```

This generates `data/processed/multitask_dataset.npz` with:
- Features: (N, 30, 25) - 30-frame windows of 25 joint angles
- Classification labels: (N,) - Exercise type (0-4)
- Segmentation labels: (N, 30) - Per-frame repetition probability

### 5. Train Model

Train the model:

```bash
python train.py
```

This creates `models/main/main.keras` with the trained model.

### 6. Run Demos

#### Video Analysis Demo
Analyze a video and generate results:

```bash
python src/demo/demo.py --video data/test_videos/test0.mp4 --output demo_output
```

#### Live Demo
Real-time exercise detection from webcam:

```bash
cd src/demo/live
./start_live_demo.sh
```

## 📊 Dataset Details

### Multitask Dataset Creation

The `multitask_dataset.npz` is created by:

1. **Feature Extraction**: MediaPipe pose detection → 25 joint angles per frame
2. **Temporal Windowing**: 30-frame sliding windows (1 second at 30 FPS)
3. **Label Augmentation**: Gaussian smoothing around repetition markers
4. **Class Balancing**: Includes no-exercise samples for better generalization

### Label Augmentation

Instead of binary 0/1 labels, we use Gaussian augmentation:
- Center (rep start): Probability = 1.0
- ±4 frames: Probability = 0.5  
- ±12 frames: Probability ≈ 0.1
- Creates smooth temporal patterns for better training

### Dataset Statistics

- **Total Sequences**: ~46,700
- **Window Size**: 30 frames (1 second)
- **Features**: 25 joint angles per frame
- **Classes**: 5 (push-ups, squats, pull-ups, dips, no-exercise)
- **Positive Samples**: ~9.9% (repetition frames)

## 🧠 Model Details

### Architecture
- **Input**: (batch_size, 30, 25) - 30 frames × 25 joint angles
- **TCN Backbone**: 8 residual blocks with dilated convolutions
- **Attention**: Multi-head attention for global temporal dependencies
- **Outputs**: 
  - Classification: 5 classes (softmax)
  - Segmentation: 30 probabilities (sigmoid)

### Training Configuration
- **Optimizer**: Adam (lr=5e-4)
- **Loss**: Focal Loss (γ=1.0, α=0.5) + Binary Crossentropy
- **Balanced Sampling**: 20% positive, 80% negative samples
- **Augmentation**: Gaussian label smoothing
- **Regularization**: Dropout (0.25), Early stopping (patience=20)

## 🎮 Demo Applications

### 1. Video Analysis Demo (`src/demo/demo.py`)

Analyzes pre-recorded videos and generates:
- Annotated video with detected repetitions
- Analysis plot showing exercise classification and repetition detection
- Repetition count and confidence scores

**Usage:**
```bash
python src/demo/demo.py --video path/to/video.mp4 --output output_directory
```

### 2. Live Demo (`src/demo/live/`)

Real-time exercise detection from webcam:
- Live pose detection and skeleton overlay
- Real-time repetition counting
- Exercise type classification

**Usage:**
```bash
cd src/demo/live
./start_live_demo.sh

## 📈 Performance

### Current Model Metrics
- **Classification Accuracy**: ~99.9%
- **Segmentation Precision**: ~97%
- **Segmentation Recall**: ~38%
- **AUC**: ~0.93

### Model Comparison
- **Robust Model**: Better generalization, fewer false positives
- **Gaussian Filtered**: Improved temporal consistency
- **Improved Recall**: Better detection of repetitions

## 🔧 Configuration

### Dataset Building
Edit `build_dataset.py` to modify:
- Window size (default: 30 frames)
- Gaussian augmentation parameters
- No-exercise ratio

### Model Training
Edit `src/training/trainer.py` to modify:
- Model architecture (filters, layers, dropout)
- Loss function parameters
- Training hyperparameters

### Demo Settings
Edit `src/demo/demo.py` to modify:
- Model path
- Output format
- Visualization settings

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
