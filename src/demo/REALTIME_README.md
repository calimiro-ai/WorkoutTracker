# Real-time Multitask Demo

This directory contains real-time versions of the multitask TCN demo optimized for live processing.

## Files

- `multitask_realtime_demo.py` - Real-time video processing with 3 FPS
- `multitask_camera_demo.py` - Live camera demo with 3 FPS processing
- `multitask_video_demo.py` - Original full-speed video processing

## Real-time Optimizations

### Processing Efficiency
- **Target FPS**: 3 FPS processing (configurable)
- **Frame Skipping**: Automatically calculates optimal frame skip based on input FPS
- **Time-based Processing**: Only processes frames at specified intervals
- **Reduced Peak Detection**: More sensitive parameters for real-time responsiveness

### Performance Benefits
- **Faster Processing**: ~10x faster than full-speed processing
- **Lower CPU Usage**: Reduced computational load
- **Real-time Capable**: Suitable for live camera feeds
- **Maintained Accuracy**: Still provides reliable exercise detection and rep counting

## Usage

### Real-time Video Processing
```bash
python3 src/demo/multitask_realtime_demo.py \
  --model models/multitask/tcn_multitask_v2_with_no_exercise/production_model.keras \
  --metadata models/multitask/tcn_multitask_v2_with_no_exercise/production_metadata.json \
  --input test_videos/test1.mp4 \
  --output output_videos/test1_realtime_3fps.mp4 \
  --fps 3
```

### Live Camera Demo
```bash
python3 src/demo/multitask_camera_demo.py \
  --model models/multitask/tcn_multitask_v2_with_no_exercise/production_model.keras \
  --metadata models/multitask/tcn_multitask_v2_with_no_exercise/production_metadata.json \
  --camera 0 \
  --fps 3 \
  --save \
  --output camera_demo.mp4
```

## Parameters

### Real-time Video Demo
- `--model`: Path to multitask model (required)
- `--metadata`: Path to metadata file (optional)
- `--input`: Input video path (required)
- `--output`: Output video path (required)
- `--fps`: Target processing FPS (default: 3)
- `--quality`: Output quality - high/medium/low (default: high)

### Camera Demo
- `--model`: Path to multitask model (required)
- `--metadata`: Path to metadata file (optional)
- `--camera`: Camera device ID (default: 0)
- `--fps`: Target processing FPS (default: 3)
- `--save`: Save video output (flag)
- `--output`: Output video path (if --save is used)

## Camera Demo Controls

- **'q'**: Quit the demo
- **'r'**: Reset rep counts
- **Ctrl+C**: Force quit

## Performance Comparison

| Demo Type | Processing FPS | Speed | Use Case |
|-----------|----------------|-------|----------|
| Original | 30 FPS | Full | High-quality analysis |
| Real-time | 3 FPS | ~10x faster | Live processing |
| Camera | 3 FPS | Live | Real-time camera feed |

## Technical Details

### Frame Processing Logic
1. Extract features from every frame
2. Only run model inference at target FPS intervals
3. Use time-based processing to maintain consistent FPS
4. Apply real-time optimized peak detection

### Peak Detection Optimization
- Reduced distance parameter for faster detection
- More sensitive prominence threshold
- Smaller width requirement
- Optimized for real-time responsiveness

### Memory Management
- Limited history buffers (100 frames max)
- Efficient deque-based storage
- Automatic cleanup of old data

## Output

Both demos provide:
- Real-time exercise classification
- Live rep counting with visual feedback
- Performance metrics overlay
- Exercise confidence display
- Processing efficiency statistics

The real-time demos are perfect for:
- Live workout tracking
- Real-time feedback systems
- Camera-based applications
- Mobile/embedded deployment
- Performance-constrained environments
