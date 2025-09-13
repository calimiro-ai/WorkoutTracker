# Multitask TCN Video Demo

This demo system creates annotated videos showing the multitask TCN model's real-time predictions for both exercise classification and repetition detection.

## Features

### 🎥 **Rich Video Overlays**
- **Exercise Classification**: Real-time exercise type detection with confidence scores
- **Rep Detection**: Per-frame repetition probability with visual threshold indicator
- **Rep Counting**: Live rep counters for each exercise type
- **Exercise Segments**: Timeline showing exercise transitions
- **Performance Metrics**: Processing stats and timing information

### 📊 **Visual Elements**
- **Color-coded exercise names** with confidence backgrounds
- **Probability bars** showing rep detection confidence  
- **Rep detection flash** when reps are detected
- **Mini timeline** at bottom showing exercise segments
- **Real-time graph** of recent repetition probabilities
- **Professional overlay design** with transparency

### 💾 **Output Files**
- **Annotated MP4 videos** with full overlay information
- **Detailed JSON results** with frame-by-frame analysis
- **Exercise segmentation data** with precise timing
- **Processing statistics** and performance metrics

## Usage

### Basic Usage
```bash
python3 src/demo/multitask_video_demo.py \
    --model models/multitask/tcn_multitask_v1/production_model.keras \
    --input test_videos/your_video.mp4 \
    --output output_videos/demo.mp4
```

### Advanced Options
```bash
python3 src/demo/multitask_video_demo.py \
    --model models/multitask/tcn_multitask_v1/production_model.keras \
    --input test_videos/test1.mp4 \
    --output output_videos/test1_demo.mp4 \
    --metadata models/multitask/tcn_multitask_v1/production_metadata.json \
    --fps 30 \
    --quality high \
    --save-results
```

### Command Line Options

- `--model`: Path to trained multitask model (.keras file)
- `--input`: Path to input video file
- `--output`: Path for output video (optional, auto-generated if not specified)
- `--metadata`: Path to model metadata JSON file (optional)
- `--fps`: Target FPS for output video (default: 30)
- `--quality`: Output quality - 'high', 'medium', or 'low' (default: high)
- `--save-results`: Save detailed processing results to JSON file

## Examples

### Create High-Quality Demo
```bash
python3 src/demo/multitask_video_demo.py \
    --model models/multitask/tcn_multitask_v1/production_model.keras \
    --input test_videos/test1.mp4 \
    --fps 30 \
    --quality high \
    --save-results
```

### Create Compressed Demo
```bash
python3 src/demo/multitask_video_demo.py \
    --model models/multitask/tcn_multitask_v1/production_model.keras \
    --input test_videos/test2.mp4 \
    --fps 15 \
    --quality medium
```

## Output Structure

For input `test_video.mp4`, the demo creates:

### Video Output
- `test_video_demo.mp4` - Annotated video with overlays

### JSON Results (if --save-results used)
- `test_video_demo_results.json` - Detailed analysis containing:
  - Processing statistics
  - Rep counts by exercise
  - Exercise segments with timing
  - Video information and metadata

## Overlay Information

### Main Panel (Top)
- **Title**: "Multitask TCN - Exercise Detection & Rep Counting"
- **Current Time**: Video timestamp
- **Exercise Detection**: Current exercise with colored background
- **Confidence Score**: Classification confidence percentage
- **Rep Probability**: Current repetition probability with color-coded bar
- **Threshold Indicator**: White line showing rep detection threshold

### Rep Counters (Right)
- **Live Counters**: Rep counts for each exercise type
- **Color Coding**: Each exercise has its distinct color

### Timeline (Bottom)
- **Exercise Segments**: Colored bars showing exercise transitions
- **Current Position**: White line indicating current time
- **Full Video Duration**: Spans entire video length

### Rep History Graph (Bottom Right)
- **Real-time Plot**: Recent repetition probability history
- **Threshold Line**: Visual rep detection threshold
- **Trend Visualization**: Shows probability patterns over time

## Technical Details

### Processing Pipeline
1. **Feature Extraction**: MediaPipe pose detection → 25 joint angles
2. **Sliding Window**: 30-frame windows for model input
3. **Multitask Prediction**: Single forward pass for both tasks
4. **Rep Detection**: Threshold-based peak detection
5. **Overlay Rendering**: Real-time annotation with rich graphics

### Performance
- **Model Size**: 747KB (191K parameters)
- **Processing Speed**: Depends on video resolution and hardware
- **Memory Usage**: ~100MB during processing
- **Output Quality**: Maintains input resolution (configurable)

### Requirements
- **Python 3.8+**
- **TensorFlow 2.x**
- **OpenCV (cv2)**
- **NumPy**
- **MediaPipe** (for pose detection)

## Customization

The demo system is highly customizable:

### Visual Styling
- Colors for each exercise type
- Overlay transparency and positioning
- Font sizes and thickness scaling
- Bar chart dimensions and styling

### Detection Parameters
- Rep detection threshold (default: 0.7)
- Minimum gap between reps (default: 15 frames)
- History window sizes for visualization

### Output Options
- Video resolution scaling
- Frame rate adjustment  
- Quality compression levels
- Metadata inclusion

## Example Results

### Test Video 1 (26.9s)
- **Processed Frames**: 775/807
- **Exercise Segments**: 22 distinct segments
- **Primary Exercises**: 65% dips, 21% squats, 14% pull-ups
- **Processing Time**: ~172 seconds
- **Output Size**: ~12MB (high quality)

### Test Video 0 (5.0s)  
- **Processed Frames**: 121/150
- **Exercise Segments**: 1 main segment
- **Primary Exercise**: 100% dips
- **Processing Time**: ~30 seconds
- **Output Size**: ~3.5MB (medium quality)

## Integration

The demo system can be easily integrated into larger workflows:

### Batch Processing
```bash
for video in test_videos/*.mp4; do
    python3 src/demo/multitask_video_demo.py \
        --model models/multitask/tcn_multitask_v1/production_model.keras \
        --input "$video" \
        --quality medium \
        --save-results
done
```

### Automated Analysis
The JSON output files can be processed for:
- Exercise recognition accuracy analysis
- Rep counting validation
- Performance benchmarking
- Dataset annotation verification

## Troubleshooting

### Common Issues
1. **Missing output directory**: The script creates directories automatically
2. **Large processing times**: Use lower quality settings or reduced FPS
3. **Model not found**: Ensure the model path is correct
4. **MediaPipe errors**: Update MediaPipe to latest version

### Performance Optimization
- Use `--quality medium` or `--quality low` for faster processing
- Reduce `--fps` for shorter videos
- Process shorter video segments for testing

This demo system showcases the power of the multitask TCN model in a visually compelling way, making it perfect for presentations, validation, and analysis of the exercise detection system! 