"""
Module for extracting pose data from cricket batting images.
Adapts the video-based PoseExtractor to work with image datasets.
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
import glob

class ImagePoseExtractor:
    """Extract pose landmarks from images of cricket batting."""
    
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
            static_image_mode=True,  # Set to True for images
            model_complexity=1,  # Reduced complexity for better detection
            min_detection_confidence=0.3,  # Lower confidence threshold
            min_tracking_confidence=0.3
        )
    
    def process_image(self, image_path, output_dir=None, visualize=False):
        """
        Process an image file to extract pose landmarks.
        
        Args:
            image_path (str): Path to the image file
            output_dir (str): Directory to save pose data
            visualize (bool): Whether to save visualization images
            
        Returns:
            dict: Pose landmarks for the image
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent / 'processed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to open image file: {image_path}")
        
        print(f"Image loaded: {image.shape}")
        
        # Prepare output
        pose_data = {
            'image_name': image_path.name,
            'landmarks': None
        }
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"RGB image shape: {rgb_image.shape}")
        
        results = self.pose.process(rgb_image)
        print(f"MediaPipe results: {results.pose_landmarks is not None}")
        
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
            pose_data['landmarks'] = landmarks
            
            # Visualize if requested
            if visualize:
                vis_dir = Path(output_dir) / 'visualizations'
                os.makedirs(vis_dir, exist_ok=True)
                
                annotated_image = image.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                cv2.imwrite(str(vis_dir / f"{image_path.stem}_pose.jpg"), annotated_image)
        
        # Save to JSON
        output_path = Path(output_dir) / f"{image_path.stem}_pose.json"
        with open(output_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        return pose_data, output_path
    
    def batch_process(self, input_dir, output_dir=None, file_extensions=None, class_label=None, limit=None):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Directory with image files
            output_dir (str): Directory to save pose data
            file_extensions (list): List of file extensions to process
            class_label (str): Optional class label for the images
            limit (int): Optional limit on number of images to process
            
        Returns:
            list: List of paths to processed data files
            list: List of dictionaries with image paths and class labels
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg']
            
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / 'processed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        metadata = []
        
        # Get all image files
        all_images = []
        for ext in file_extensions:
            all_images.extend(list(input_dir.glob(f'*{ext}')))
            all_images.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        # Limit the number of images if specified
        if limit and limit > 0:
            all_images = all_images[:limit]
        
        for image_path in all_images:
            print(f"Processing {image_path.name}...")
            try:
                _, output_path = self.process_image(image_path, output_dir)
                output_files.append(str(output_path))
                
                # Add metadata
                metadata.append({
                    'image_path': str(image_path),
                    'pose_path': str(output_path),
                    'class': class_label or input_dir.name  # Use directory name as class if not provided
                })
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        # Save metadata
        metadata_path = Path(output_dir) / f"{input_dir.name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_files, metadata
    
    def process_dataset(self, dataset_dir, output_dir=None, file_extensions=None, limit_per_class=None):
        """
        Process a dataset with multiple class directories.
        
        Args:
            dataset_dir (str): Root directory of the dataset
            output_dir (str): Directory to save pose data
            file_extensions (list): List of file extensions to process
            limit_per_class (int): Optional limit on number of images per class
            
        Returns:
            dict: Dictionary with class names as keys and metadata as values
        """
        dataset_dir = Path(dataset_dir)
        if output_dir is None:
            output_dir = dataset_dir.parent / 'processed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_metadata = []
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            print(f"\nProcessing class: {class_dir.name}")
            class_output_dir = Path(output_dir) / class_dir.name
            _, metadata = self.batch_process(
                class_dir, 
                output_dir=class_output_dir,
                file_extensions=file_extensions,
                class_label=class_dir.name,
                limit=limit_per_class
            )
            all_metadata.extend(metadata)
        
        # Save combined metadata
        combined_metadata_path = Path(output_dir) / "dataset_metadata.json"
        with open(combined_metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\nDataset processing complete. Processed {len(all_metadata)} images across {len(class_dirs)} classes.")
        return all_metadata


if __name__ == "__main__":
    # Example usage
    config_path = "../../configs/config.yaml"
    extractor = ImagePoseExtractor(config_path)
    
    # Process a single image
    # extractor.process_image("../../data/drive/drives1.png", visualize=True)
    
    # Or batch process a directory
    # extractor.batch_process("../../data/drive")
    
    # Or process the entire dataset
    # extractor.process_dataset("../../data")