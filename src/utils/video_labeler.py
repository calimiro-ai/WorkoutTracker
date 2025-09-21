#!/usr/bin/env python3
"""
Simple Video Labeling Tool

This script helps you manually label videos for training data collection.
It loops through videos in a specified folder and allows you to mark frames
where exercises occur by pressing Enter.

Usage:
    python src/utils/video_labeler.py --input_folder data/raw --output_folder data/labels
"""

import os
import sys
import argparse
import cv2
import pandas as pd
from pathlib import Path
import json
import time

class VideoLabeler:
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
        
        # Filter out videos that already have labels and exclude no-exercise videos
        self.unlabeled_videos = []
        for video_path in self.video_files:
            # Skip no-exercise videos - they should be handled separately
            if 'no-exercise' in str(video_path):
                print(f"Skipping no-exercise video: {video_path.relative_to(self.input_folder)}")
                continue
                
            # Create corresponding label path with subfolder structure
            relative_path = video_path.relative_to(self.input_folder)
            
            # Check for multiple label formats:
            # 1. Old format: video.csv
            # 2. New format: video.csv
            # 3. Any CSV file with similar name
            old_label_file = self.output_folder / relative_path.parent / f"{relative_path.stem}.csv"
            new_label_file = self.output_folder / relative_path.parent / f"{relative_path.stem}.csv"
            
            # Check if any CSV file exists for this video
            label_exists = False
            if old_label_file.exists():
                label_exists = True
                print(f"Found old format labels: {old_label_file}")
            elif new_label_file.exists():
                label_exists = True
                print(f"Found new format labels: {new_label_file}")
            else:
                # Check for any CSV file with similar name in the same directory
                label_dir = self.output_folder / relative_path.parent
                if label_dir.exists():
                    for csv_file in label_dir.glob(f"{relative_path.stem}*.csv"):
                        label_exists = True
                        print(f"Found existing labels: {csv_file}")
                        break
            
            if not label_exists:
                self.unlabeled_videos.append(video_path)
        
        print(f"Found {len(self.video_files)} total video files")
        print(f"Found {len(self.unlabeled_videos)} unlabeled videos to process (excluding no-exercise)")
        
        if len(self.unlabeled_videos) == 0:
            print("All videos are already labeled!")
        else:
            print("\nUnlabeled videos:")
            for video in self.unlabeled_videos:
                print(f"  - {video.relative_to(self.input_folder)}")
    
    def label_video(self, video_path):
        """Label a single video file."""
        print(f"\n{'='*60}")
        print(f"LABELING: {video_path.name}")
        print(f"Path: {video_path.relative_to(self.input_folder)}")
        print(f"{'='*60}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print("\nInstructions:")
        print("- Video plays continuously")
        print("- Press ENTER to mark current frame as exercise (label=1)")
        print("- Press 'p' to pause/unpause")
        print("- Press 'f' to skip 300 frames forward")
        print("- Press 'b' to skip 300 frames backward")
        print("- Press 'q' to quit and save")
        print("- Press 's' to skip to specific frame")
        
        # Initialize labels array
        labels = [0] * total_frames
        current_frame = 0
        is_paused = False
        video_completed = False
        
        # Create window
        window_name = f"Labeling: {video_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Calculate frame delay for smooth playback (30 FPS)
        frame_delay = int(1000 / fps)  # milliseconds
        
        try:
            while True:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video reached")
                    video_completed = True
                    break
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # Display frame info
                frame_info = f"Frame: {current_frame}/{total_frames} | Label: {labels[current_frame]} | Time: {current_frame/fps:.2f}s"
                cv2.putText(display_frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show current label status
                label_color = (0, 255, 0) if labels[current_frame] == 1 else (0, 0, 255)
                label_text = "EXERCISE" if labels[current_frame] == 1 else "NO EXERCISE"
                cv2.putText(display_frame, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                
                # Show pause status
                pause_text = "PAUSED" if is_paused else "PLAYING"
                pause_color = (0, 0, 255) if is_paused else (0, 255, 0)
                cv2.putText(display_frame, pause_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pause_color, 2)
                
                # Show instructions
                cv2.putText(display_frame, "ENTER: mark exercise | p: pause | f: +300 frames | b: -300 frames | s: skip | q: quit", 
                           (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display_frame)
                
                # Handle key presses with timeout for continuous playback
                if is_paused:
                    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely when paused
                else:
                    key = cv2.waitKey(frame_delay) & 0xFF  # Wait for frame delay when playing
                
                if key == ord('q'):  # Quit
                    print("Video labeling interrupted by user")
                    video_completed = False
                    break
                elif key == 13:  # Enter - mark as exercise
                    labels[current_frame] = 1
                    print(f"*** MARKED FRAME {current_frame} AS EXERCISE ***")
                elif key == ord('p'):  # Pause/unpause
                    is_paused = not is_paused
                    print(f"Video {'paused' if is_paused else 'resumed'}")
                elif key == ord('f'):  # Skip 300 frames forward
                    current_frame = min(current_frame + 300, total_frames - 1)
                    print(f"Skipped to frame {current_frame}")
                elif key == ord('b'):  # Skip 300 frames backward
                    current_frame = max(current_frame - 300, 0)
                    print(f"Skipped to frame {current_frame}")
                elif key == ord('s'):  # Skip to frame
                    try:
                        target_frame = int(input(f"Enter frame number (0-{total_frames-1}): "))
                        current_frame = max(0, min(target_frame, total_frames - 1))
                    except ValueError:
                        print("Invalid frame number")
                
                # Advance frame if not paused
                if not is_paused:
                    current_frame += 1
                    if current_frame >= total_frames:
                        print("End of video reached")
                        video_completed = True
                        break
        
        except KeyboardInterrupt:
            print("Video labeling interrupted by user")
            video_completed = False
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
        
        # Only save labels if video was completed
        if video_completed:
            self.save_labels(video_path, labels, fps)
            return True
        else:
            print("Video not saved due to interruption")
            return False
    
    def save_labels(self, video_path, labels, fps):
        """Save labels to CSV file in corresponding subfolder."""
        # Create corresponding label path with subfolder structure
        relative_path = video_path.relative_to(self.input_folder)
        # Use new format with _labels suffix
        label_file = self.output_folder / relative_path.parent / f"{relative_path.stem}.csv"
        
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
        
        # Save metadata in same location
        metadata_file = label_file.parent / f"{relative_path.stem}_metadata.json"
        metadata = {
            'video_path': str(video_path),
            'total_frames': len(labels),
            'fps': fps,
            'exercise_frames': sum(labels),
            'exercise_percentage': sum(labels) / len(labels) * 100
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nLabels saved to: {label_file}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Exercise frames: {sum(labels)}/{len(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    
    def run(self):
        """Run the labeling process for all unlabeled videos."""
        if not self.unlabeled_videos:
            print("No unlabeled videos found!")
            return
        
        print(f"Starting video labeling process...")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        
        for i, video_path in enumerate(self.unlabeled_videos, 1):
            print(f"\nProcessing video {i}/{len(self.unlabeled_videos)}")
            success = self.label_video(video_path)
            
            if success:
                print(f"Video {i} completed successfully")
            else:
                print(f"Video {i} was interrupted - not saved")
            
            # Automatically continue to next video (no y/n prompt)
            if i < len(self.unlabeled_videos):
                print(f"Automatically continuing to next video...")
                time.sleep(1)  # Brief pause between videos
        
        print(f"\nLabeling process completed!")
        print(f"Check output folder: {self.output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Video Labeling Tool')
    parser.add_argument('--input_folder', default='data/raw', 
                       help='Folder containing videos to label')
    parser.add_argument('--output_folder', default='data/labels', 
                       help='Folder to save label files')
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder not found: {args.input_folder}")
        return 1
    
    try:
        labeler = VideoLabeler(args.input_folder, args.output_folder)
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
