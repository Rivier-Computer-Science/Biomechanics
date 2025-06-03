"""
Main cricket biomechanics analysis module.
Handles the end-to-end pipeline from video to feedback.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import cv2
import torch
import time

from ..data_collection.pose_extractor import PoseExtractor
from ..feature_engineering.joint_angles import JointAngleCalculator
from ..model.lstm_model import StraightDriveClassifier
from ..visualization.pose_visualizer import PoseVisualizer


class CricketTechniqueAnalyzer:
    """End-to-end cricket batting technique analysis."""
    
    def __init__(self, config_path=None, model_path=None):
        """
        Initialize the cricket technique analyzer.
        
        Args:
            config_path (str): Path to the configuration file
            model_path (str): Path to the trained model
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'pose': {'model': 'mediapipe'},
                'model': {'sequence_length': 60}
            }
        
        # Initialize components
        self.pose_extractor = PoseExtractor(config_path)
        self.angle_calculator = JointAngleCalculator()
        self.visualizer = PoseVisualizer(config_path)
        self.classifier = StraightDriveClassifier(config_path)
        
        # Load model if provided
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            try:
                # We need input size and num_classes to load the model
                # In a real implementation, you would save these with the model
                self.classifier.load_model(model_path, input_size=120, num_classes=2)
                self.model_loaded = True
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def analyze_video(self, video_path, output_dir=None, visualize=True):
        """
        Analyze a cricket batting video.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save results
            visualize (bool): Whether to create visualization
            
        Returns:
            dict: Analysis results including technique feedback
        """
        start_time = time.time()
        
        # Set up output directory
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = video_path.parent / 'analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Analyzing video: {video_path.name}")
        
        # Step 1: Extract pose data
        print("Step 1/4: Extracting pose data...")
        pose_data = self.pose_extractor.process_video(
            video_path, 
            output_dir=output_dir,
            visualize=visualize
        )
        pose_path = Path(output_dir) / f"{video_path.stem}_pose.json"
        
        # Step 2: Calculate joint angles and features
        print("Step 2/4: Calculating biomechanical features...")
        features_df = self.angle_calculator.process_pose_data(pose_path)
        features_df = self.angle_calculator.calculate_dynamic_features(features_df)
        features_path = Path(output_dir) / f"{video_path.stem}_features.csv"
        features_df.to_csv(features_path, index=False)
        
        # Step 3: Segment into cricket phases (simplified approach)
        # In a real implementation, you would use a more sophisticated method
        print("Step 3/4: Segmenting into cricket phases...")
        total_frames = len(features_df)
        features_df['phase'] = 'unknown'
        
        # Simple heuristic: divide into 4 equal phases
        quarter = total_frames // 4
        features_df.loc[0:quarter, 'phase'] = 'backlift'
        features_df.loc[quarter+1:quarter*2, 'phase'] = 'downswing'
        features_df.loc[quarter*2+1:quarter*3, 'phase'] = 'impact'
        features_df.loc[quarter*3+1:, 'phase'] = 'follow_through'
        
        # Step 4: Classify technique and generate feedback
        print("Step 4/4: Analyzing technique...")
        
        # Prepare sequences for the model
        if self.model_loaded:
            seq_length = self.config['model'].get('sequence_length', 60)
            
            # Get sequences (exclude phase column)
            feature_cols = [col for col in features_df.columns if col not in ['frame_idx', 'timestamp', 'phase']]
            sequences = []
            
            for i in range(0, len(features_df) - seq_length + 1, seq_length // 2):  # 50% overlap
                end_idx = min(i + seq_length, len(features_df))
                if end_idx - i < seq_length // 2:  # Skip if too short
                    continue
                    
                seq = features_df.iloc[i:end_idx][feature_cols].values
                
                # Pad if necessary
                if seq.shape[0] < seq_length:
                    padding = np.zeros((seq_length - seq.shape[0], seq.shape[1]))
                    seq = np.vstack([seq, padding])
                
                sequences.append(seq)
            
            if sequences:
                sequences = np.array(sequences)
                
                # Make predictions
                predictions, probabilities = self.classifier.predict(sequences)
                
                # Map predictions to technique assessment
                assessment = []
                for i, pred in enumerate(predictions):
                    confidence = probabilities[i, pred]
                    
                    if pred == 1:  # Assuming 1 is correct technique
                        assessment.append({
                            'segment': i,
                            'technique': 'correct',
                            'confidence': float(confidence),
                            'feedback': 'Good technique in this segment.'
                        })
                    else:
                        assessment.append({
                            'segment': i,
                            'technique': 'incorrect',
                            'confidence': float(confidence),
                            'feedback': 'Technique needs improvement in this segment.'
                        })
            else:
                assessment = [{
                    'segment': 0,
                    'technique': 'unknown',
                    'confidence': 0.0,
                    'feedback': 'Not enough frames to analyze.'
                }]
        else:
            # If model not loaded, provide placeholder assessment
            assessment = [{
                'segment': 0,
                'technique': 'not_evaluated',
                'confidence': 0.0,
                'feedback': 'Model not loaded. Technique classification unavailable.'
            }]
        
        # Generate visual report
        if visualize:
            # Add technique score to features dataframe (placeholder)
            features_df['technique_score'] = 75  # Example score
            
            # Create annotated video
            video_output = self.visualizer.create_phase_annotation_video(
                str(video_path),
                str(pose_path),
                features_df,
                str(Path(output_dir) / f"{video_path.stem}_analyzed.mp4")
            )
            
            # Create angle plot
            angle_cols = [col for col in features_df.columns if 'angle' in col]
            plot_path = Path(output_dir) / f"{video_path.stem}_angles.png"
            self.visualizer.create_angle_time_series_plot(
                features_df,
                angle_cols,
                str(plot_path)
            )
            
            # Generate technique report
            report = self.visualizer.generate_technique_report(
                features_df,
                str(Path(output_dir) / f"{video_path.stem}_report.json")
            )
        
        # Prepare results
        analysis_duration = time.time() - start_time
        
        results = {
            'video_name': video_path.name,
            'duration': float(analysis_duration),
            'total_frames': total_frames,
            'output_dir': str(output_dir),
            'phase_breakdown': {
                phase: len(features_df[features_df['phase'] == phase])
                for phase in features_df['phase'].unique()
            },
            'technique_assessment': assessment
        }
        
        # Save results
        results_path = Path(output_dir) / f"{video_path.stem}_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete. Results saved to {results_path}")
        return results
    
    def analyze_batch(self, input_dir, output_dir=None, file_extensions=None):
        """
        Analyze all videos in a directory.
        
        Args:
            input_dir (str): Directory with video files
            output_dir (str): Directory to save results
            file_extensions (list): List of file extensions to process
            
        Returns:
            list: List of analysis results
        """
        if file_extensions is None:
            file_extensions = ['.mp4', '.avi', '.mov']
            
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / 'analysis'
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for ext in file_extensions:
            for video_path in input_dir.glob(f'*{ext}'):
                print(f"\nProcessing {video_path.name}...")
                try:
                    result = self.analyze_video(
                        str(video_path),
                        output_dir=output_dir
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {video_path.name}: {e}")
        
        # Save batch results
        batch_results = {
            'processed_videos': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        batch_path = Path(output_dir) / 'batch_analysis.json'
        with open(batch_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Example usage
    config_path = "../../configs/config.yaml"
    analyzer = CricketTechniqueAnalyzer(config_path)
    
    # Analyze a single video
    # analyzer.analyze_video("../../data/raw/sample_drive.mp4")
    
    # Or batch process a directory
    # analyzer.analyze_batch("../../data/raw")
