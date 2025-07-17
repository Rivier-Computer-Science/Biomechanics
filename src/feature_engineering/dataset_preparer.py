"""
Module for preparing cricket shot dataset for training.
Extracts features from pose data and organizes them for the classifier.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from ..feature_engineering.joint_angles import JointAngleCalculator


class CricketShotDatasetPreparer:
    """Prepare cricket shot dataset for training."""
    
    def __init__(self, config_path=None):
        """
        Initialize the dataset preparer.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.angle_calculator = JointAngleCalculator()
        self.config_path = config_path
    
    def process_pose_files(self, metadata_file, output_dir=None):
        """
        Process pose files to extract features.
        
        Args:
            metadata_file (str): Path to the metadata JSON file
            output_dir (str): Directory to save feature files
            
        Returns:
            list: List of dictionaries with image paths, feature paths, and class labels
        """
        metadata_path = Path(metadata_file)
        if output_dir is None:
            output_dir = metadata_path.parent / 'features'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        processed_metadata = []
        
        for item in metadata:
            pose_path = item['pose_path']
            class_label = item['class']
            
            try:
                # Load pose data
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                
                # Skip if no landmarks
                if not pose_data.get('landmarks'):
                    print(f"No landmarks found in {pose_path}, skipping...")
                    continue
                
                # Adapt pose data format for the angle calculator
                # The angle calculator expects a specific format with frames
                adapted_pose_data = {
                    'frames': [{
                        'frame_idx': 0,
                        'timestamp': 0,
                        'landmarks': pose_data['landmarks']
                    }]
                }
                
                # Create a temporary JSON file
                temp_pose_path = Path(output_dir) / f"temp_{Path(pose_path).stem}.json"
                with open(temp_pose_path, 'w') as f:
                    json.dump(adapted_pose_data, f)
                
                # Calculate joint angles
                features_df = self.angle_calculator.process_pose_data(temp_pose_path)
                
                # Add class label
                features_df['class'] = class_label
                
                # Save features
                feature_path = Path(output_dir) / f"{Path(pose_path).stem}_features.csv"
                features_df.to_csv(feature_path, index=False)
                
                # Add to processed metadata
                processed_item = item.copy()
                processed_item['feature_path'] = str(feature_path)
                processed_metadata.append(processed_item)
                
                # Clean up temporary file
                os.remove(temp_pose_path)
                
            except Exception as e:
                print(f"Error processing {pose_path}: {e}")
        
        # Save processed metadata
        processed_metadata_path = Path(output_dir) / f"processed_{metadata_path.stem}.json"
        with open(processed_metadata_path, 'w') as f:
            json.dump(processed_metadata, f, indent=2)
        
        return processed_metadata
    
    def combine_features(self, metadata_list, output_file):
        """
        Combine features from multiple files into a single dataset.
        
        Args:
            metadata_list (list): List of dictionaries with feature paths and class labels
            output_file (str): Path to save the combined dataset
            
        Returns:
            pd.DataFrame: Combined features dataframe
        """
        all_features = []
        
        for item in metadata_list:
            feature_path = item['feature_path']
            class_label = item['class']
            
            try:
                # Load features
                features_df = pd.read_csv(feature_path)
                
                # Ensure class label is present
                if 'class' not in features_df.columns:
                    features_df['class'] = class_label
                
                all_features.append(features_df)
                
            except Exception as e:
                print(f"Error loading {feature_path}: {e}")
        
        if not all_features:
            raise ValueError("No feature files could be loaded")
        
        # Combine all features
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Save combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"Combined dataset saved to {output_path}")
        return combined_df
    
    def prepare_train_test_split(self, features_df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            features_df (pd.DataFrame): Combined features dataframe
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Ensure class column is present
        if 'class' not in features_df.columns:
            raise ValueError("Class column not found in features dataframe")
        
        # Check if we have enough samples for splitting
        class_counts = features_df['class'].value_counts()
        min_samples = class_counts.min()
        num_classes = len(class_counts)
        
        print(f"Class distribution: {class_counts.to_dict()}")
        print(f"Minimum samples per class: {min_samples}")
        print(f"Number of classes: {num_classes}")
        
        # Calculate minimum required samples for splitting
        min_test_samples = max(1, int(test_size * len(features_df) / num_classes))
        min_val_samples = max(1, int(val_size * len(features_df) / num_classes))
        min_required = min_test_samples + min_val_samples + 1  # +1 for training
        
        print(f"Minimum required samples per class: {min_required}")
        
        # Check if we can do stratified split
        total_samples = len(features_df)
        test_samples = int(test_size * total_samples)
        val_samples = int(val_size * total_samples / (1 - test_size))
        
        print(f"Total samples: {total_samples}")
        print(f"Test samples needed: {test_samples}")
        print(f"Val samples needed: {val_samples}")
        
        # Need at least as many samples as classes for stratified split
        if min_samples < min_required or test_samples < num_classes or val_samples < num_classes:
            print(f"WARNING: Not enough samples for proper splitting. Using simple split without stratification.")
            # Use simple random split without stratification
            train_val_df, test_df = train_test_split(
                features_df,
                test_size=test_size,
                random_state=random_state
            )
            
            # Adjust validation size to account for the first split
            adjusted_val_size = val_size / (1 - test_size)
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_size,
                random_state=random_state
            )
        else:
            # Use stratified split as originally intended
            train_val_df, test_df = train_test_split(
                features_df,
                test_size=test_size,
                stratify=features_df['class'],
                random_state=random_state
            )
            
            # Second split: training vs validation
            # Adjust validation size to account for the first split
            adjusted_val_size = val_size / (1 - test_size)
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_size,
                stratify=train_val_df['class'],
                random_state=random_state
            )
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df, val_df, test_df, output_dir):
        """
        Save the dataset splits to disk.
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            val_df (pd.DataFrame): Validation dataframe
            test_df (pd.DataFrame): Test dataframe
            output_dir (str): Directory to save the splits
            
        Returns:
            dict: Paths to the saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save class mapping
        classes = sorted(train_df['class'].unique())
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        
        mapping_path = output_dir / "class_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        paths = {
            'train_path': str(train_path),
            'val_path': str(val_path),
            'test_path': str(test_path),
            'class_mapping': str(mapping_path)
        }
        
        print(f"Dataset splits saved to {output_dir}")
        print(f"Class mapping: {class_mapping}")
        
        return paths


if __name__ == "__main__":
    # Example usage
    config_path = "../../configs/config.yaml"
    preparer = CricketShotDatasetPreparer(config_path)
    
    # Process pose files
    # metadata = preparer.process_pose_files("../../data/processed/dataset_metadata.json")
    
    # Combine features
    # features_df = preparer.combine_features(metadata, "../../data/processed/combined_features.csv")
    
    # Prepare train/test split
    # train_df, val_df, test_df = preparer.prepare_train_test_split(features_df)
    
    # Save splits
    # preparer.save_splits(train_df, val_df, test_df, "../../data/processed/splits")