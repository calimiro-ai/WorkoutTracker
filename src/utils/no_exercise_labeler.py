#!/usr/bin/env python3
"""
No-Exercise Video Labeler

Automatically labels all videos in the no-exercise folder with label=0 for all frames.
This script processes videos that should have no exercise content.

Usage:
    python src/utils/no_exercise_labeler.py --input_folder data/raw/no-exercise --output_folder data/labels/no_exercise
"""

import os
import sys
import argparse
import cv2
import pandas as pd
from pathlib import Path
import json

class NoExerciseLabeler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all video files recursively
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        self.video_files = []
        
        # Recursively find all video files
        for ext in self.video_extensions:
            self.video_files.extend(self.input_folder.rglob(f'*{ext}'))
            self.video_files.extend(self.input_folder.rglob(f'*{ext.upper()}'))
        
        # Filter out videos that already have labels
        self.unlabeled_videos = []
        for video_path in self.video_files:
            # Create corresponding label path
            relative_path = video_path.relative_to(self.input_folder)
            label_file = self.output_folder / f"{relative_path.stem}_labels.csv"
            
            if not label_file.exists():
                self.unlabeled_videos.append(video_path)
        
        print(f"Found {len(self.video_files)} total no-exercise video files")
        print(f"Found {len(self.unlabeled_videos)} unlabeled no-exercise videos to process")
        
        if len(self.unlabeled_videos) == 0:
            print("All no-exercise videos are already labeled!")
        else:
            print("\nUnlabeled no-exercise videos:")
            for video in self.unlabeled_videos:
                print(f"  - {video.relative_to(self.input_folder)}")
    
    def label_video(self, video_path):
        """Automatically label a no-exercise video with all 0s."""
        print(f"\n{'='*60}")
        print(f"PROCESSING NO-EXERCISE: {video_path.name}")
        print(f"Path: {video_path.relative_to(self.input_folder)}")
        print(f"{'='*60}")
        
        # Open video to get properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print("Automatically labeling all frames as NO EXERCISE (label=0)")
        
        # Create labels array with all 0s
        labels = [0] * total_frames
        
        # Save labels
        self.save_labels(video_path, labels, fps)
        
        # Cleanup
        cap.release()
    
    def save_labels(self, video_path, labels, fps):
        """Save labels to CSV file."""
        # Create label file path
        relative_path = video_path.relative_to(self.input_folder)
        label_file = self.output_folder / f"{relative_path.stem}_labels.csv"
        
        # Create parent directories if they don't exist
        label_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'frame': range(len(labels)),
            'time': [i/fps for i in range(len(labels))],
            'label': labels
        })
        
        # Save CSV
        df.to_csv(label_file, index=False)
        
        # Save metadata
        metadata_file = label_file.parent / f"{relative_path.stem}_metadata.json"
        metadata = {
            'video_path': str(video_path),
            'total_frames': len(labels),
            'fps': fps,
            'exercise_frames': sum(labels),
            'exercise_percentage': sum(labels) / len(labels) * 100,
            'video_type': 'no-exercise'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Labels saved to: {label_file}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"All frames labeled as NO EXERCISE (0/{len(labels)} = 0.0%)")
    
    def run(self):
        """Run the automatic labeling process for all unlabeled no-exercise videos."""
        if not self.unlabeled_videos:
            print("No unlabeled no-exercise videos found!")
            return
        
        print(f"Starting automatic no-exercise labeling...")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        
        for i, video_path in enumerate(self.unlabeled_videos, 1):
            print(f"\nProcessing video {i}/{len(self.unlabeled_videos)}")
            self.label_video(video_path)
        
        print(f"\nNo-exercise labeling completed!")
        print(f"All videos automatically labeled with label=0")


def main():
    parser = argparse.ArgumentParser(description='No-Exercise Video Labeler')
    parser.add_argument('--input_folder', default='data/raw/no-exercise', 
                       help='Folder containing no-exercise videos')
    parser.add_argument('--output_folder', default='data/labels/no_exercise', 
                       help='Folder to save label files')
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder not found: {args.input_folder}")
        return 1
    
    try:
        labeler = NoExerciseLabeler(args.input_folder, args.output_folder)
        labeler.run()
        return 0
    except KeyboardInterrupt:
        print("\nLabeling interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
