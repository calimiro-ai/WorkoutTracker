# Utilities

This directory contains utility tools for data processing and video analysis.

## 📁 Contents

### Core Utilities

- **`video_labeler.py`**: Interactive video labeling tool
- **`csv_format_converter.py`**: Label format conversion
- **`no_exercise_labeler.py`**: Automated no-exercise labeling

## 🏷️ Video Labeling Tool

### Interactive Labeling

**Usage:**
```bash
python src/utils/video_labeler.py path/to/video.mp4
```

**Features:**
- Frame-by-frame video playback
- Click to mark repetition starts
- Keyboard shortcuts for efficiency
- Real-time label preview
- Automatic CSV export

**Keyboard Controls:**
- `Space`: Play/pause video
- `Left/Right`: Navigate frames
- `Click`: Mark repetition start
- `R`: Reset current labels
- `S`: Save labels to CSV
- `Q`: Quit without saving

### Label Format

**CSV Output:**
```csv
frame,label
0,0
1,0
2,1
3,0
...
```

**Label Values:**
- `0`: No repetition
- `1`: Repetition start marker

## 🔄 Format Conversion

### CSV Converter

**Usage:**
```bash
python src/utils/csv_format_converter.py input.csv output.csv
```

**Features:**
- Convert between different label formats
- Validate label consistency
- Merge multiple label files
- Export to different formats

### Supported Formats

1. **Binary Format**: 0/1 labels per frame
2. **Probability Format**: 0.0-1.0 probabilities
3. **Interval Format**: Start/end frame pairs
4. **Metadata Format**: JSON with additional info

## 🚫 No-Exercise Labeling

### Automated Labeling

**Usage:**
```bash
python src/utils/no_exercise_labeler.py data/raw/no-exercise/
```

**Features:**
- Batch process no-exercise videos
- Generate all-zero labels automatically
- Create metadata files
- Validate video content

**Output:**
- CSV files with all-zero labels
- Metadata JSON files
- Processing logs

## 🛠️ Data Processing Tools

### Video Validation

**Check video properties:**
```python
from src.utils.video_validator import validate_video
validate_video('path/to/video.mp4')
```

**Validation checks:**
- Video format compatibility
- Frame rate consistency
- Resolution requirements
- Duration limits

### Label Validation

**Check label consistency:**
```python
from src.utils.label_validator import validate_labels
validate_labels('path/to/labels.csv', video_frames=900)
```

**Validation checks:**
- Frame count alignment
- Label value ranges
- Temporal consistency
- Missing data detection

## 📊 Analysis Tools

### Label Statistics

**Analyze label distribution:**
```python
from src.utils.label_analyzer import analyze_labels
stats = analyze_labels('path/to/labels.csv')
print(f"Total repetitions: {stats['total_reps']}")
print(f"Average rep length: {stats['avg_rep_length']}")
```

**Statistics provided:**
- Total repetition count
- Average repetition length
- Repetition frequency
- Label density

### Video Analysis

**Extract video metadata:**
```python
from src.utils.video_analyzer import analyze_video
info = analyze_video('path/to/video.mp4')
print(f"Duration: {info['duration']}s")
print(f"FPS: {info['fps']}")
print(f"Resolution: {info['width']}x{info['height']}")
```

## 🔧 Customization

### Labeling Interface

**Modify `video_labeler.py`:**
```python
# Custom keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    'space': 'play_pause',
    'left': 'prev_frame',
    'right': 'next_frame',
    'click': 'mark_rep',
    'r': 'reset_labels'
}
```

### Format Conversion

**Add new formats:**
```python
class CustomFormatConverter:
    def convert_to_custom(self, labels):
        # Custom conversion logic
        pass
    
    def convert_from_custom(self, data):
        # Custom parsing logic
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Video Playback**: Check codec compatibility
2. **Label Saving**: Verify file permissions
3. **Format Errors**: Validate input format
4. **Memory Issues**: Process videos in batches

### Debug Mode

Enable verbose output:
```bash
python src/utils/video_labeler.py video.mp4 --verbose
```

### Error Handling

**Graceful error recovery:**
- Automatic backup creation
- Error logging and reporting
- Partial save on interruption
- Data validation checks

## 📚 Advanced Usage

### Batch Processing

**Process multiple videos:**
```bash
for video in data/raw/push-ups/*.mp4; do
    python src/utils/video_labeler.py "$video"
done
```

### Custom Workflows

**Integration with other tools:**
```python
from src.utils.video_labeler import VideoLabeler
from src.utils.csv_format_converter import CSVConverter

# Label video
labeler = VideoLabeler('video.mp4')
labeler.run()

# Convert format
converter = CSVConverter()
converter.convert('labels.csv', 'formatted_labels.csv')
```

### Quality Assurance

**Automated quality checks:**
```python
from src.utils.quality_checker import QualityChecker

checker = QualityChecker()
if checker.validate_dataset('data/labels/'):
    print("All labels passed quality checks")
else:
    print("Some labels need attention")
```
