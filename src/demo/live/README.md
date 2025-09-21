# Live Demo

Real-time exercise detection and counting from webcam input.

## 🚀 Quick Start

```bash
cd src/demo/live
./start_live_demo.sh
```

## 📁 Contents

- **`live_demo.py`**: Main live demo application
- **`start_live_demo.sh`**: Startup script
- **`requirements.txt`**: Live demo dependencies

## 🎯 Features

### Real-time Processing
- Live pose detection at 30 FPS
- Real-time repetition counting
- Exercise type classification
- Visual feedback with skeleton overlay

### Interactive Controls
- **Space**: Pause/resume detection
- **R**: Reset repetition counter
- **Q**: Quit application
- **C**: Toggle confidence display

### Visual Feedback
- Skeleton overlay on detected poses
- Repetition counter display
- Exercise type indicator
- Confidence scores
- Status messages

## 🔧 Configuration

### Model Settings
Edit `live_demo.py` to modify:
- Model path and loading
- Detection thresholds
- Display settings
- Performance parameters

### Camera Settings
```python
# Camera configuration
camera_id = 0              # Camera device ID
resolution = (640, 480)    # Video resolution
fps = 30                   # Target frame rate
```

### Detection Parameters
```python
# Detection thresholds
min_confidence = 0.5       # Minimum pose confidence
rep_threshold = 0.7        # Repetition detection threshold
smooth_factor = 0.8        # Temporal smoothing
```

## 📊 Performance

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB+ available memory
- **Camera**: USB webcam or built-in camera
- **OS**: Linux, macOS, or Windows

### Optimization Tips
1. **Close other applications** to free up resources
2. **Use lower resolution** for better performance
3. **Ensure good lighting** for pose detection
4. **Position camera** at appropriate distance

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Found**: Check camera connection and permissions
2. **Poor Performance**: Reduce resolution or close other apps
3. **Detection Errors**: Improve lighting and camera positioning
4. **Model Loading**: Ensure model file exists and is accessible

### Debug Mode

Enable verbose output:
```bash
python live_demo.py --verbose
```

### Performance Monitoring

Check system resources:
```bash
# Monitor CPU usage
htop

# Monitor GPU usage (if available)
nvidia-smi
```

## 🎮 Usage Tips

### Best Practices
1. **Good Lighting**: Ensure adequate lighting for pose detection
2. **Clear Background**: Minimize background distractions
3. **Appropriate Distance**: Position camera 2-3 meters away
4. **Stable Position**: Keep camera steady during exercise

### Exercise Guidelines
1. **Full Range of Motion**: Complete full repetitions
2. **Consistent Form**: Maintain proper exercise form
3. **Clear Visibility**: Ensure all body parts are visible
4. **Smooth Movements**: Avoid jerky or rapid motions

## 🔧 Customization

### Adding New Exercises
1. Train model with new exercise data
2. Update exercise type mapping
3. Adjust detection parameters
4. Test with live demo

### Custom Visualizations
```python
# Custom overlay colors
skeleton_color = (0, 255, 0)      # Green skeleton
text_color = (255, 255, 255)      # White text
background_color = (0, 0, 0)      # Black background
```

### Performance Tuning
```python
# Adjust processing parameters
skip_frames = 1                    # Process every N frames
buffer_size = 10                   # Frame buffer size
max_queue_size = 100              # Maximum queue size
```

## 📚 Integration

### With Other Applications
- **Fitness Apps**: Integrate with workout tracking
- **Gaming**: Use for exercise-based games
- **Education**: Teaching proper exercise form
- **Research**: Data collection for studies

### API Usage
```python
from live_demo import LiveExerciseDetector

detector = LiveExerciseDetector()
detector.start()

# Get real-time results
while detector.is_running():
    result = detector.get_latest_result()
    if result:
        print(f"Exercise: {result['exercise_type']}")
        print(f"Reps: {result['rep_count']}")
```

## 🎯 Future Enhancements

### Planned Features
- Multiple exercise detection
- Form analysis and feedback
- Workout session tracking
- Social sharing capabilities
- Mobile app integration

### Technical Improvements
- GPU acceleration support
- Multi-camera support
- Cloud processing options
- Advanced pose analysis
- Real-time coaching
