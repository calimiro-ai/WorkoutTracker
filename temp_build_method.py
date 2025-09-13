    def build(self, balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Build the complete multitask dataset with 3 FPS processing.
        """
        print("Building multitask dataset with 3 FPS processing...")
        
        all_sequences = []
        all_cls_labels = []
        all_seg_labels = []
        samples_per_exercise = {}
        
        # Process each exercise type
        for exercise_idx, exercise_type in enumerate(self.exercise_types):
            print(f"\nProcessing {exercise_type}...")
            
            try:
                # Get video files for this exercise
                exercise_dir = os.path.join(self.videos_dir, exercise_type)
                if not os.path.exists(exercise_dir):
                    print(f"  Directory not found: {exercise_dir}")
                    continue
                    
                video_files = [f for f in os.listdir(exercise_dir) if f.endswith('.mp4')]
                print(f"  Found {len(video_files)} videos for {exercise_type}")
                
                if not video_files:
                    print(f"  No videos found for {exercise_type}")
                    continue
                
                # Process videos directly
                exercise_sequences = []
                exercise_cls_labels = []
                exercise_seg_labels = []
                
                for video_file in video_files[:5]:  # Limit to first 5 videos for testing
                    video_path = os.path.join(exercise_dir, video_file)
                    print(f"    Processing {video_file}...")
                    
                    # Extract features from video
                    features = self._extract_video_features(video_path)
                    if features is None or len(features) == 0:
                        print(f"    No features extracted from {video_file}")
                        continue
                    
                    # Create dummy segmentation labels (all zeros for now)
                    seg_labels = np.zeros(len(features))
                    
                    # Apply 3 FPS processing
                    features_3fps, seg_labels_3fps = self._apply_3fps_processing(features, seg_labels)
                    
                    # Create windowed sequences
                    sequences, cls_labels, seg_window_labels = self._create_windowed_sequences(
                        features_3fps, seg_labels_3fps, exercise_idx
                    )
                    
                    if len(sequences) > 0:
                        exercise_sequences.extend(sequences)
                        exercise_cls_labels.extend(cls_labels)
                        exercise_seg_labels.extend(seg_window_labels)
                        print(f"    Created {len(sequences)} sequences from {video_file}")
                
                if len(exercise_sequences) > 0:
                    all_sequences.extend(exercise_sequences)
                    all_cls_labels.extend(exercise_cls_labels)
                    all_seg_labels.extend(exercise_seg_labels)
                    samples_per_exercise[exercise_type] = len(exercise_sequences)
                    print(f"  Total sequences for {exercise_type}: {len(exercise_sequences)}")
                else:
                    print(f"  No sequences created for {exercise_type}")
                    
            except Exception as e:
                print(f"  Error processing {exercise_type}: {e}")
                continue
        
        if len(all_sequences) == 0:
            raise ValueError("No valid sequences created from any exercise type")
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y_classification = np.array(all_cls_labels)
        y_segmentation = np.array(all_seg_labels)
        
        print(f"\nDataset built successfully:")
        print(f"  Total sequences: {len(X)}")
        print(f"  Sequence shape: {X.shape}")
        print(f"  Classification labels shape: {y_classification.shape}")
        print(f"  Segmentation labels shape: {y_segmentation.shape}")
        print(f"  Samples per exercise: {samples_per_exercise}")
        
        # Create metadata
        metadata = {
            'exercise_types': self.exercise_types,
            'num_classes': len(self.exercise_types),
            'samples_per_exercise': samples_per_exercise,
            'target_fps': self.target_fps,
            'window_size': self.window_size,
            'stride': self.stride
        }
        
        return X, y_classification, y_segmentation, metadata
