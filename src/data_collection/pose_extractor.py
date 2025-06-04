"""
Module for extracting pose data from cricket batting videos.
Uses MediaPipe for pose estimation.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
import yaml
import time

class PoseExtractor:
    """Extract pose landmarks from videos of cricket batting."""
    
    def __init__(self, config_path=None):
        """
        Initialize the pose extractor with configurations.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'pose': {
                    'model': 'mediapipe',
                    'confidence_threshold': 0.5
                }
            }
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=self.config['pose'].get('confidence_threshold', 0.5),
            min_tracking_confidence=0.5
        )
    
    def process_video(self, video_path, output_dir=None, visualize=False):
        """
        Process a video file to extract pose landmarks.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save pose data
            visualize (bool): Whether to save visualization frames
            
        Returns:
            dict: Pose landmarks for each frame
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = video_path.parent / 'processed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output
        pose_data = {
            'video_name': video_path.name,
            'fps': fps,
            'frame_count': frame_count,
            'frames': []
        }
        
        if visualize:
            vis_dir = Path(output_dir) / 'visualizations' / video_path.stem
            os.makedirs(vis_dir, exist_ok=True)
        
        # Process each frame
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'landmarks': None
            }
            
            if results.pose_landmarks:
                # Convert landmarks to list
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['landmarks'] = landmarks
                
                # Visualize if requested
                if visualize:
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    cv2.imwrite(str(vis_dir / f"frame_{frame_idx:04d}.jpg"), annotated_frame)
            
            pose_data['frames'].append(frame_data)
            frame_idx += 1
        
        cap.release()
        
        # Save to JSON
        output_path = Path(output_dir) / f"{video_path.stem}_pose.json"
        with open(output_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        return pose_data
    
    def batch_process(self, input_dir, output_dir=None, file_extensions=None):
        """
        Process all videos in a directory.
        
        Args:
            input_dir (str): Directory with video files
            output_dir (str): Directory to save pose data
            file_extensions (list): List of file extensions to process
            
        Returns:
            list: List of paths to processed data files
        """
        if file_extensions is None:
            file_extensions = ['.mp4', '.avi', '.mov']
            
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / 'processed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        for ext in file_extensions:
            for video_path in input_dir.glob(f'*{ext}'):
                print(f"Processing {video_path.name}...")
                try:
                    pose_data = self.process_video(video_path, output_dir)
                    output_path = Path(output_dir) / f"{video_path.stem}_pose.json"
                    output_files.append(str(output_path))
                except Exception as e:
                    print(f"Error processing {video_path.name}: {e}")
        
        return output_files


if __name__ == "__main__":
    # Example usage
    config_path = "../../configs/config.yaml"
    extractor = PoseExtractor(config_path)
    
    # Process a single video
    # extractor.process_video("../../data/raw/sample_drive.mp4", visualize=True)
    
    # Or batch process a directory
    # extractor.batch_process("../../data/raw")
