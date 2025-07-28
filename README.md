# Calimiro AI - Real-time Exercise Detection

A computer vision system for real-time exercise classification and repetition counting using MediaPipe pose detection and deep learning models.

## 🏗️ Project Structure

```
SmartMirror/
├── 📁 src/                        Source code
│   ├── 📁 backend_interface/
│   │   ├── backend_interface.py   Backend interface to communicate with the mirror
│   │   ├── server.py              HTTP server to receive data batches from the mirror
│   │   └── shared_data.py         Thread-Safe IPC
│   ├── 📁 core/                   Core functionality
│   │   ├── realtime_pipeline.py   Main realtime pipeline
│   │   ├── dataset_builder.py     Dataset creation and management
│   │   ├── realtime.py            Realtime model & OpenCV rendering
│   │   ├── realtime_no_counting.py
│   │   ├── main.py                Main script
│   │   └── __init__.py
│   ├── 📁 training/               Model training
│   │   ├── train_classifier.py    Classification model training
│   │   ├── retrain_segmentation_realtime.py   Real-time segmentation training
│   │   ├── classification/        Classification modules
│   │   └── segmentation/          Segmentation modules
│   ├── 📁 testing/                Testing and debugging
│   │   ├── test_*.py              Various test scripts
│   │   ├── debug_classification.py
│   │   └── inspect_classification_dataset.py
│   ├── 📁 utils/                  Utility scripts
│   │   ├── classification_demo.py
│   │   ├── segmentation_demo.py
│   │   ├── mediapipe_demo.py
│   │   └── video_labeler.py
│   └── __init__.py
├── 📁 data/                       Data storage
│   ├── 📁 raw/                    Raw video files
│   ├── 📁 processed/              Processed datasets
│   └── 📁 labels/                 Annotation files
├── 📁 models/                     Trained models
│   ├── 📁 classification/         Classification models
│   └── 📁 segmentation/           Segmentation models
├── 📁 logs/                       Training logs
├── 📁 output_videos/              Output videos
├── 📁 test_videos/                Test videos
├── 📁 docs/                       Documentation
├── 📁 scripts/                    Utility scripts
├── 🐍 run_realtime.py             Main entry point
├── 📄 requirements.txt            Dependencies
└── 📄 README.md                   This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SmartMirror
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the real-time pipeline:**
   ```bash
   python run_realtime.py
   ```

### Usage

- **Press 'q'** to quit the application
- **Press 'r'** to reset rep counts
- The system will automatically detect and classify exercises
- Rep counts are displayed in real-time

## 🧠 Features

### ✅ Exercise Classification
- **4 Exercise Types**: Push-ups, Squats, Pull-ups, Dips
- **Real-time Detection**: Live webcam feed processing
- **High Accuracy**: 99.98% test accuracy
- **Low Latency**: Optimized for real-time performance

### ✅ Rep Counting
- **State Machine**: Advanced rep detection algorithm
- **Low-FPS Optimized**: Works with 3 FPS performance
- **Multiple Exercises**: Simultaneous tracking
- **Visual Feedback**: Real-time state display

### ✅ Performance Optimizations
- **30-Frame Windows**: Optimized for real-time processing
- **Frame Skipping**: Reduces computational load
- **Adaptive Thresholds**: Adjusts to performance constraints
- **Efficient Models**: Lightweight neural networks

## 🔧 Development

### Training Models

**Classification Model:**
```bash
cd src/training
python train_classifier.py
```

**Real-time Segmentation Models:**
```bash
cd src/training
python retrain_segmentation_realtime.py
```

### Testing

**Test with video files:**
```bash
cd src/testing
python test_realtime_video.py test_videos/test0.mp4
```

**Debug classification:**
```bash
cd src/testing
python debug_classification.py test_videos/test0.mp4
```

## 📊 Performance

- **Classification Accuracy**: 99.98%
- **Real-time FPS**: ~3 FPS (CPU-based)
- **Latency**: ~350ms per frame
- **Memory Usage**: ~2GB RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Pose detection framework
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library 