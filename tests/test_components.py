"""
Basic tests to verify that the cricket biomechanics analysis components work together.
"""

import sys
import os
import unittest
import numpy as np
import json
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.pose_extractor import PoseExtractor
from src.feature_engineering.joint_angles import JointAngleCalculator
from src.model.lstm_model import StraightDriveClassifier
from src.visualization.pose_visualizer import PoseVisualizer
from src.inference.cricket_analyzer import CricketTechniqueAnalyzer


class TestComponentIntegration(unittest.TestCase):
    """Test the integration of components."""
    
    def setUp(self):
        """Set up test data and directories."""
        self.test_dir = Path(os.path.dirname(__file__)) / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a mock config file for testing
        self.config_path = self.test_dir / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            f.write("""
            # Test configuration
            pose:
              model: mediapipe
              confidence_threshold: 0.5
            
            model:
              type: lstm
              sequence_length: 10
              hidden_units: 64
              dropout: 0.3
              learning_rate: 0.001
            """)
        
        # Create mock pose data for testing
        self.pose_data = {
            'video_name': 'test_video.mp4',
            'fps': 30.0,
            'frame_count': 10,
            'frames': []
        }
        
        # Generate 10 frames with 33 landmarks each (MediaPipe standard)
        for i in range(10):
            landmarks = []
            for j in range(33):
                landmarks.append({
                    'x': 0.5 + 0.1 * np.sin(i * 0.1 + j * 0.05),
                    'y': 0.5 + 0.1 * np.cos(i * 0.1 + j * 0.05),
                    'z': 0.1 * np.sin(i * 0.1),
                    'visibility': 0.9
                })
            
            self.pose_data['frames'].append({
                'frame_idx': i,
                'timestamp': i / 30.0,
                'landmarks': landmarks
            })
        
        # Save mock pose data
        self.pose_path = self.test_dir / 'test_pose.json'
        with open(self.pose_path, 'w') as f:
            json.dump(self.pose_data, f)
    
    def test_pose_extractor_initialization(self):
        """Test that PoseExtractor can be initialized."""
        extractor = PoseExtractor(str(self.config_path))
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(extractor.pose)
    
    def test_joint_angle_calculator(self):
        """Test that JointAngleCalculator can process pose data."""
        calculator = JointAngleCalculator()
        
        # Process the mock pose data
        df = calculator.process_pose_data(str(self.pose_path))
        
        # Check that we have a DataFrame with the expected columns
        self.assertIsNotNone(df)
        self.assertTrue('frame_idx' in df.columns)
        self.assertTrue('timestamp' in df.columns)
        
        # Check for angle calculations
        angle_columns = [col for col in df.columns if 'angle' in col]
        self.assertTrue(len(angle_columns) > 0)
    
    def test_classifier_initialization(self):
        """Test that StraightDriveClassifier can be initialized."""
        classifier = StraightDriveClassifier(str(self.config_path))
        self.assertIsNotNone(classifier)
    
    def test_visualizer_initialization(self):
        """Test that PoseVisualizer can be initialized."""
        visualizer = PoseVisualizer(str(self.config_path))
        self.assertIsNotNone(visualizer)
    
    def test_analyzer_initialization(self):
        """Test that CricketTechniqueAnalyzer can be initialized."""
        analyzer = CricketTechniqueAnalyzer(str(self.config_path))
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(analyzer.pose_extractor)
        self.assertIsNotNone(analyzer.angle_calculator)
        self.assertIsNotNone(analyzer.visualizer)
        self.assertIsNotNone(analyzer.classifier)
    
    def tearDown(self):
        """Clean up test files."""
        # Note: In a real test, you might want to clean up files
        # but for this demonstration, we'll leave them for inspection
        pass


if __name__ == '__main__':
    unittest.main()
