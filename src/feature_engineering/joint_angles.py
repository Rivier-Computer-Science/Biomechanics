"""
Module for calculating joint angles and other biomechanical features from pose data.
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path


def calculate_vector(point1, point2):
    """
    Calculate vector between two points.
    
    Args:
        point1 (dict): First point with x, y, z coordinates
        point2 (dict): Second point with x, y, z coordinates
        
    Returns:
        np.array: Vector from point1 to point2
    """
    return np.array([
        point2['x'] - point1['x'],
        point2['y'] - point1['y'],
        point2['z'] - point1['z']
    ])


def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors in degrees.
    
    Args:
        v1 (np.array): First vector
        v2 (np.array): Second vector
        
    Returns:
        float: Angle in degrees
    """
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Handle potential numerical errors
    cos_angle = min(max(dot_product / (norm_product + 1e-10), -1.0), 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * 180 / np.pi
    
    return angle_deg


class JointAngleCalculator:
    """Calculate joint angles from pose landmarks."""
    
    def __init__(self):
        """Initialize the joint angle calculator."""
        # Define joint indices for MediaPipe pose landmarks
        # Based on MediaPipe's documentation
        self.landmark_indices = {
            # Torso
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            
            # Arms
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            
            # Legs
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
    def process_pose_data(self, pose_data_path):
        """
        Process pose data to calculate joint angles.
        
        Args:
            pose_data_path (str): Path to the pose data JSON file
            
        Returns:
            pd.DataFrame: DataFrame with frame indices and calculated angles
        """
        # Load pose data
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)
        
        frames = []
        for frame in pose_data['frames']:
            if frame['landmarks'] is None:
                continue
                
            landmarks = frame['landmarks']
            
            # Calculate angles
            angles = {
                'frame_idx': frame['frame_idx'],
                'timestamp': frame['timestamp']
            }
            
            # Calculate elbow angles (shoulder-elbow-wrist)
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['left_shoulder', 'left_elbow', 'left_wrist']):
                left_upper_arm = calculate_vector(
                    landmarks[self.landmark_indices['left_shoulder']],
                    landmarks[self.landmark_indices['left_elbow']]
                )
                left_forearm = calculate_vector(
                    landmarks[self.landmark_indices['left_elbow']],
                    landmarks[self.landmark_indices['left_wrist']]
                )
                angles['left_elbow_angle'] = calculate_angle(left_upper_arm, left_forearm)
            
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['right_shoulder', 'right_elbow', 'right_wrist']):
                right_upper_arm = calculate_vector(
                    landmarks[self.landmark_indices['right_shoulder']],
                    landmarks[self.landmark_indices['right_elbow']]
                )
                right_forearm = calculate_vector(
                    landmarks[self.landmark_indices['right_elbow']],
                    landmarks[self.landmark_indices['right_wrist']]
                )
                angles['right_elbow_angle'] = calculate_angle(right_upper_arm, right_forearm)
            
            # Calculate knee angles (hip-knee-ankle)
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['left_hip', 'left_knee', 'left_ankle']):
                left_thigh = calculate_vector(
                    landmarks[self.landmark_indices['left_hip']],
                    landmarks[self.landmark_indices['left_knee']]
                )
                left_shin = calculate_vector(
                    landmarks[self.landmark_indices['left_knee']],
                    landmarks[self.landmark_indices['left_ankle']]
                )
                angles['left_knee_angle'] = calculate_angle(left_thigh, left_shin)
            
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['right_hip', 'right_knee', 'right_ankle']):
                right_thigh = calculate_vector(
                    landmarks[self.landmark_indices['right_hip']],
                    landmarks[self.landmark_indices['right_knee']]
                )
                right_shin = calculate_vector(
                    landmarks[self.landmark_indices['right_knee']],
                    landmarks[self.landmark_indices['right_ankle']]
                )
                angles['right_knee_angle'] = calculate_angle(right_thigh, right_shin)
            
            # Calculate shoulder angles (torso-shoulder-elbow)
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['left_hip', 'left_shoulder', 'left_elbow']):
                left_torso = calculate_vector(
                    landmarks[self.landmark_indices['left_hip']],
                    landmarks[self.landmark_indices['left_shoulder']]
                )
                left_upper_arm = calculate_vector(
                    landmarks[self.landmark_indices['left_shoulder']],
                    landmarks[self.landmark_indices['left_elbow']]
                )
                angles['left_shoulder_angle'] = calculate_angle(left_torso, left_upper_arm)
            
            if all(self.landmark_indices[joint] < len(landmarks) for joint in ['right_hip', 'right_shoulder', 'right_elbow']):
                right_torso = calculate_vector(
                    landmarks[self.landmark_indices['right_hip']],
                    landmarks[self.landmark_indices['right_shoulder']]
                )
                right_upper_arm = calculate_vector(
                    landmarks[self.landmark_indices['right_shoulder']],
                    landmarks[self.landmark_indices['right_elbow']]
                )
                angles['right_shoulder_angle'] = calculate_angle(right_torso, right_upper_arm)
            
            frames.append(angles)
        
        # Create DataFrame
        df = pd.DataFrame(frames)
        return df
    
    def calculate_dynamic_features(self, df, window_size=5):
        """
        Calculate dynamic features like velocities and accelerations.
        
        Args:
            df (pd.DataFrame): DataFrame with joint angles
            window_size (int): Window size for smoothing
            
        Returns:
            pd.DataFrame: DataFrame with added dynamic features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate velocities (degrees per second)
        angle_columns = [col for col in df.columns if col.endswith('_angle')]
        for col in angle_columns:
            velocity_col = col.replace('_angle', '_velocity')
            result_df[velocity_col] = df[col].diff() / df['timestamp'].diff()
            
            # Apply smoothing (rolling average)
            result_df[velocity_col] = result_df[velocity_col].rolling(window=window_size, center=True).mean()
            
            # Calculate acceleration (degrees per second^2)
            accel_col = col.replace('_angle', '_acceleration')
            result_df[accel_col] = result_df[velocity_col].diff() / df['timestamp'].diff()
            result_df[accel_col] = result_df[accel_col].rolling(window=window_size, center=True).mean()
        
        return result_df
    
    def save_features(self, df, output_path):
        """
        Save the calculated features to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame with calculated features
            output_path (str): Path to save the CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    calculator = JointAngleCalculator()
    
    # Process a single pose data file
    # df = calculator.process_pose_data("../../data/processed/sample_drive_pose.json")
    # df_with_dynamics = calculator.calculate_dynamic_features(df)
    # calculator.save_features(df_with_dynamics, "../../data/processed/sample_drive_features.csv")
