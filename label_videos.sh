#!/bin/bash
# Video Labeling Helper Script

echo "=== Video Labeling Tools ==="
echo "1. Label exercise videos (manual labeling)"
echo "2. Process no-exercise videos (automatic labeling)"
echo "3. Check labeling status"
echo "4. Exit"
echo

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "Starting manual video labeling..."
        echo "Put your exercise videos in data/raw/"
        echo "Press ENTER to mark frames with exercises"
        echo "Only unlabeled videos will be processed automatically"
        echo
        python src/utils/video_labeler.py --input_folder data/raw --output_folder data/labels
        ;;
    2)
        echo "Processing no-exercise videos..."
        echo "Put your no-exercise videos in data/no_exercise/"
        echo "All frames will be automatically labeled as 0"
        echo "Only unlabeled videos will be processed automatically"
        echo
        python src/utils/no_exercise_labeler.py --input_folder data/no_exercise --output_folder data/labels
        ;;
    3)
        echo "Checking labeling status..."
        echo
        
        # Check exercise videos
        if [ -d "data/raw" ]; then
            total_exercise=$(find data/raw -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.wmv" | wc -l)
            labeled_exercise=$(find data/labels -name "*_labels.csv" | wc -l)
            echo "Exercise videos: $labeled_exercise/$total_exercise labeled"
        else
            echo "No data/raw directory found"
        fi
        
        # Check no-exercise videos
        if [ -d "data/no_exercise" ]; then
            total_no_exercise=$(find data/no_exercise -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.wmv" | wc -l)
            echo "No-exercise videos: $total_no_exercise total"
        else
            echo "No data/no_exercise directory found"
        fi
        
        echo
        echo "Label files in data/labels/:"
        ls -la data/labels/ 2>/dev/null || echo "No labels directory found"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
