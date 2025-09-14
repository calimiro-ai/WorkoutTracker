#!/usr/bin/env python3
"""
Run Realtime Demo on All Test Videos

Runs the realtime demo with saved plots on all test videos and generates
comprehensive analysis plots and data for each video.
"""

import os
import sys
import subprocess
import glob

def main():
    """Run realtime demo on all test videos."""
    
    # Configuration
    model_path = "models/multitask_tcn_30fps/best_model.keras"
    test_videos_dir = "data/test_videos"
    output_dir = "output_videos"
    target_fps = 30
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all test videos
    video_patterns = ["*.mp4", "*.avi", "*.mov"]
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(test_videos_dir, pattern)))
    
    if not video_files:
        print(f"No video files found in {test_videos_dir}")
        return
    
    video_files.sort()  # Process in order
    print(f"Found {len(video_files)} videos to process:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Process each video
    success_count = 0
    total_count = len(video_files)
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{total_count}: {video_name}")
        print(f"{'='*60}")
        
        try:
            # Run the realtime demo script
            cmd = [
                sys.executable, "src/demo/realtime_demo_save_plots.py",
                "--model", model_path,
                "--video", video_path,
                "--output", output_dir,
                "--fps", str(target_fps),
                "--quality", "high"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully processed {video_name}")
                success_count += 1
            else:
                print(f"❌ Failed to process {video_name}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout processing {video_name} (took longer than 5 minutes)")
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Output directory: {output_dir}")
    
    # List generated files
    if success_count > 0:
        print(f"\nGenerated files:")
        generated_files = sorted(glob.glob(os.path.join(output_dir, "*")))
        for file_path in generated_files:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            if file_size > 1024*1024:
                size_str = f"{file_size/(1024*1024):.1f}MB"
            elif file_size > 1024:
                size_str = f"{file_size/1024:.1f}KB"
            else:
                size_str = f"{file_size}B"
            print(f"  - {file_name} ({size_str})")


if __name__ == "__main__":
    main() 