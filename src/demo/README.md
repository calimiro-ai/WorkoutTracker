# Demo Applications

This directory contains demo applications for the WorkoutTracker system.

## 📁 Contents

### 1. Video Analysis Demo (`demo.py`)

Analyzes pre-recorded videos and generates comprehensive results including:
- Annotated video with detected repetitions
- Analysis plots showing exercise classification and repetition detection
- Repetition count and confidence scores

**Usage:**
```bash
python src/demo/demo.py --video path/to/video.mp4 --output output_directory
```

**Parameters:**
- `--video`: Path to input video file
- `--output`: Output directory for results
- `--model`: Model path (default: models/main/main.keras)
- `--window-size`: Window size for analysis (default: 30)
- `--stride`: Stride for sliding window (default: 1)

**Output Files:**
- `{video_name}_analysis.png`: Analysis plot with classification and segmentation
- `{video_name}_annotated.mp4`: Video with repetition markers

### 2. Live Demo (`live/`)

Real-time exercise detection from webcam with:
- Live pose detection and skeleton overlay
- Real-time repetition counting
- Exercise type classification
- Interactive controls

**Usage:**
```bash
cd src/demo/live
./start_live_demo.sh
```

**Features:**
- Real-time processing at 30 FPS
- Visual feedback with skeleton overlay
- Repetition counter display
- Exercise type indicator
- Pause/resume functionality

## �� Demo Workflow

1. **Prepare Model**: Ensure trained model exists in `models/main/`
2. **Choose Demo**: Select video analysis or live demo
3. **Run Analysis**: Execute appropriate demo script
4. **Review Results**: Check generated outputs and metrics

## 📊 Understanding Output

### Analysis Plot
- **Top Panel**: Exercise classification probability over time
- **Bottom Panel**: Repetition detection signal with peaks
- **Metrics**: Total repetitions, confidence scores, exercise type

### Annotated Video
- Green markers indicate detected repetition starts
- Skeleton overlay shows pose detection
- Frame counter and confidence display

## 🔧 Customization

### Model Selection
Edit the model path in demo scripts to use different trained models:
- `models/main/main.keras` - Current best model
- `models/robust/best_model.keras` - Robust model
- `models/gaussian/best_model.keras` - Gaussian filtered model

### Visualization Settings
Modify plot appearance in `demo.py`:
- Colors and styling
- Plot dimensions
- Annotation settings
- Output format

## 🐛 Troubleshooting

### Common Issues
1. **Model Not Found**: Ensure model exists and path is correct
2. **Video Format**: Use MP4 format for best compatibility
3. **Memory Issues**: Reduce window size for large videos
4. **Performance**: Use GPU acceleration if available

### Debug Mode
Enable verbose output:
```bash
python src/demo/demo.py --video video.mp4 --output output --verbose
```
