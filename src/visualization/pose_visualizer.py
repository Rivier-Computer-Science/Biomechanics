"""
Module for visualizing pose data and providing feedback on cricket technique.
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import mediapipe as mp


class PoseVisualizer:
    """Visualize pose data and provide feedback on cricket technique."""
    
    def __init__(self, config_path=None):
        """
        Initialize the pose visualizer with configurations.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Default configurations
        self.config = {
            'visualization': {
                'skeleton_color': [0, 255, 0],  # Green
                'incorrect_color': [255, 0, 0],  # Red
                'correct_color': [0, 255, 0],   # Green
                'font_size': 1,
                'line_thickness': 2
            }
        }
        
        # Load custom config if provided
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                # Update default config with custom values
                self.config.update(custom_config)
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=tuple(self.config['visualization']['skeleton_color']),
            thickness=self.config['visualization']['line_thickness']
        )
    
    def visualize_pose_on_frame(self, frame, landmarks, annotations=None):
        """
        Draw pose landmarks on a video frame.
        
        Args:
            frame (np.array): Video frame
            landmarks (list): List of pose landmarks
            annotations (dict): Optional annotations for specific joints
            
        Returns:
            np.array: Annotated frame
        """
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Convert landmarks to MediaPipe format
        mp_landmarks = self.mp_pose.PoseLandmark
        landmark_list = mp.solutions.pose.PoseLandmarks()
        
        for i, landmark in enumerate(landmarks):
            landmark_list.landmark[i].x = landmark['x']
            landmark_list.landmark[i].y = landmark['y']
            landmark_list.landmark[i].z = landmark['z']
            landmark_list.landmark[i].visibility = landmark.get('visibility', 1.0)
        
        # Draw the skeleton
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            self.drawing_spec,
            self.drawing_spec
        )
        
        # Add annotations if provided
        if annotations:
            for joint, info in annotations.items():
                # Get joint index
                if hasattr(mp_landmarks, joint.upper()):
                    idx = getattr(mp_landmarks, joint.upper())
                    
                    # Get joint position
                    x = int(landmarks[idx]['x'] * frame.shape[1])
                    y = int(landmarks[idx]['y'] * frame.shape[0])
                    
                    # Determine color based on correctness
                    if info.get('correct', True):
                        color = tuple(self.config['visualization']['correct_color'])
                    else:
                        color = tuple(self.config['visualization']['incorrect_color'])
                    
                    # Draw circle at joint position
                    cv2.circle(annotated_frame, (x, y), 8, color, -1)
                    
                    # Add feedback text if provided
                    if 'feedback' in info:
                        cv2.putText(
                            annotated_frame,
                            info['feedback'],
                            (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.config['visualization']['font_size'],
                            color,
                            2
                        )
        
        return annotated_frame
    
    def create_angle_time_series_plot(self, features_df, angle_columns, output_path=None):
        """
        Create a time series plot of joint angles.
        
        Args:
            features_df (pd.DataFrame): DataFrame with joint angles
            angle_columns (list): List of angle column names to plot
            output_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in angle_columns:
            ax.plot(features_df['timestamp'], features_df[col], label=col)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Joint Angles During Cricket Straight Drive')
        ax.legend()
        ax.grid(True)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_phase_annotation_video(self, video_path, pose_data_path, features_df, output_path=None):
        """
        Create a video with phase annotations.
        
        Args:
            video_path (str): Path to the original video
            pose_data_path (str): Path to the pose data JSON
            features_df (pd.DataFrame): DataFrame with features and phase labels
            output_path (str): Path to save the output video
            
        Returns:
            str: Path to the output video
        """
        # Load pose data
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video
        if output_path is None:
            output_path = str(Path(video_path).with_stem(Path(video_path).stem + '_annotated'))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Check if we have pose data for this frame
            if frame_idx < len(pose_data['frames']) and pose_data['frames'][frame_idx]['landmarks']:
                landmarks = pose_data['frames'][frame_idx]['landmarks']
                
                # Get corresponding features
                features_row = features_df[features_df['frame_idx'] == frame_idx]
                
                # Prepare annotations if we have phase information
                annotations = {}
                if not features_row.empty and 'phase' in features_row.columns:
                    phase = features_row['phase'].values[0]
                    
                    # Add phase information to frame
                    cv2.putText(
                        frame,
                        f"Phase: {phase}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                    
                    # Add technique score if available
                    if 'technique_score' in features_row.columns:
                        score = features_row['technique_score'].values[0]
                        cv2.putText(
                            frame,
                            f"Score: {score:.1f}",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2
                        )
                    
                    # Identify issues based on phase
                    if phase == 'backlift':
                        # Example: Check elbow angle in backlift
                        if 'right_elbow_angle' in features_row.columns:
                            elbow_angle = features_row['right_elbow_angle'].values[0]
                            if elbow_angle < 90:  # Just an example threshold
                                annotations['right_elbow'] = {
                                    'correct': False,
                                    'feedback': 'Elbow too bent'
                                }
                    
                    elif phase == 'downswing':
                        # Example: Check shoulder rotation
                        if 'right_shoulder_angle' in features_row.columns:
                            shoulder_angle = features_row['right_shoulder_angle'].values[0]
                            if shoulder_angle < 45:  # Just an example threshold
                                annotations['right_shoulder'] = {
                                    'correct': False,
                                    'feedback': 'Insufficient rotation'
                                }
                    
                    elif phase == 'impact':
                        # Example: Check wrist position
                        if 'right_wrist' in landmarks:
                            annotations['right_wrist'] = {
                                'correct': True,
                                'feedback': 'Good contact position'
                            }
                    
                    elif phase == 'follow_through':
                        # Example: Check body balance
                        if 'left_knee_angle' in features_row.columns:
                            knee_angle = features_row['left_knee_angle'].values[0]
                            if knee_angle < 130:  # Just an example threshold
                                annotations['left_knee'] = {
                                    'correct': False,
                                    'feedback': 'Knee bent too much'
                                }
                
                # Draw pose and annotations
                annotated_frame = self.visualize_pose_on_frame(frame, landmarks, annotations)
                out.write(annotated_frame)
            else:
                # Write original frame if no pose data
                out.write(frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path
    
    def generate_technique_report(self, features_df, output_path=None):
        """
        Generate a comprehensive technique report.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features and annotations
            output_path (str): Path to save the report
            
        Returns:
            dict: Technique assessment
        """
        # Initialize report
        report = {
            'overall_score': 0,
            'phases': {},
            'recommendations': []
        }
        
        # Check if we have phase information
        if 'phase' not in features_df.columns:
            return report
        
        # Analyze each phase
        phases = features_df['phase'].unique()
        
        for phase in phases:
            phase_data = features_df[features_df['phase'] == phase]
            phase_report = {'score': 0, 'issues': []}
            
            # Example: Check backlift phase
            if phase == 'backlift':
                if 'right_elbow_angle' in phase_data.columns:
                    avg_elbow = phase_data['right_elbow_angle'].mean()
                    if avg_elbow < 90:
                        phase_report['issues'].append(
                            "Elbow too bent during backlift (avg angle: {:.1f}°)".format(avg_elbow)
                        )
                        phase_report['score'] += 60
                    else:
                        phase_report['score'] += 90
                
                if 'right_shoulder_angle' in phase_data.columns:
                    avg_shoulder = phase_data['right_shoulder_angle'].mean()
                    if avg_shoulder < 45:
                        phase_report['issues'].append(
                            "Insufficient shoulder rotation (avg angle: {:.1f}°)".format(avg_shoulder)
                        )
                        phase_report['score'] += 60
                    else:
                        phase_report['score'] += 90
            
            # Example: Check downswing phase
            elif phase == 'downswing':
                # Similar checks for downswing phase
                if 'right_elbow_angle' in phase_data.columns:
                    avg_elbow = phase_data['right_elbow_angle'].mean()
                    if avg_elbow > 130:
                        phase_report['issues'].append(
                            "Elbow too straight during downswing (avg angle: {:.1f}°)".format(avg_elbow)
                        )
                        phase_report['score'] += 70
                    else:
                        phase_report['score'] += 90
            
            # Example: Check impact phase
            elif phase == 'impact':
                # Checks for impact phase
                if 'right_wrist_angle' in phase_data.columns:
                    avg_wrist = phase_data['right_wrist_angle'].mean()
                    if avg_wrist > 170:
                        phase_report['issues'].append(
                            "Wrist too stiff at impact (avg angle: {:.1f}°)".format(avg_wrist)
                        )
                        phase_report['score'] += 60
                    else:
                        phase_report['score'] += 90
            
            # Example: Check follow-through phase
            elif phase == 'follow_through':
                # Checks for follow-through phase
                if 'right_shoulder_angle' in phase_data.columns:
                    avg_shoulder = phase_data['right_shoulder_angle'].mean()
                    if avg_shoulder < 80:
                        phase_report['issues'].append(
                            "Incomplete follow-through rotation (avg angle: {:.1f}°)".format(avg_shoulder)
                        )
                        phase_report['score'] += 70
                    else:
                        phase_report['score'] += 90
            
            # Normalize phase score
            if phase_report['score'] > 0:
                phase_report['score'] = min(100, phase_report['score'])
            
            # Add to overall report
            report['phases'][phase] = phase_report
        
        # Calculate overall score
        if report['phases']:
            phase_scores = [phase_info['score'] for phase_info in report['phases'].values()]
            report['overall_score'] = sum(phase_scores) / len(phase_scores)
        
        # Generate recommendations
        for phase, phase_info in report['phases'].items():
            for issue in phase_info['issues']:
                # Add recommendations based on issues
                if "Elbow too bent" in issue:
                    report['recommendations'].append(
                        "Practice keeping a straighter arm during backlift."
                    )
                elif "Insufficient shoulder rotation" in issue:
                    report['recommendations'].append(
                        "Work on shoulder rotation exercises to improve range of motion."
                    )
                elif "Elbow too straight" in issue:
                    report['recommendations'].append(
                        "Maintain slight elbow bend during downswing for better control."
                    )
                elif "Wrist too stiff" in issue:
                    report['recommendations'].append(
                        "Practice wrist flexibility exercises to improve impact mechanics."
                    )
                elif "Incomplete follow-through" in issue:
                    report['recommendations'].append(
                        "Focus on completing the follow-through motion to improve power transfer."
                    )
        
        # Remove duplicate recommendations
        report['recommendations'] = list(set(report['recommendations']))
        
        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    # Example usage
    visualizer = PoseVisualizer("../../configs/config.yaml")
    
    # Example: Create angle time series plot
    # features_df = pd.read_csv("../../data/processed/sample_drive_features.csv")
    # visualizer.create_angle_time_series_plot(
    #     features_df,
    #     ['right_elbow_angle', 'right_shoulder_angle', 'right_knee_angle'],
    #     "../../data/processed/angle_plot.png"
    # )
    
    # Example: Create annotated video
    # visualizer.create_phase_annotation_video(
    #     "../../data/raw/sample_drive.mp4",
    #     "../../data/processed/sample_drive_pose.json",
    #     features_df,
    #     "../../data/processed/sample_drive_annotated.mp4"
    # )
