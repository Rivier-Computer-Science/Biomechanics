"""
Unit tests for joint angle calculation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering.joint_angles import JointAngleCalculator


class TestJointAngleCalculator(unittest.TestCase):
    """Test cases for JointAngleCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = JointAngleCalculator()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample pose data with proper dictionary format
        self.sample_landmarks = []
        for i in range(33):  # MediaPipe has 33 pose landmarks
            self.sample_landmarks.append({
                'x': np.random.uniform(0, 1),  # x coordinate
                'y': np.random.uniform(0, 1),  # y coordinate
                'z': np.random.uniform(0, 1),  # z coordinate
                'visibility': np.random.uniform(0.5, 1.0)  # visibility
            })
        
        self.sample_pose_data = {
            'frames': [
                {
                    'frame_idx': 0,
                    'timestamp': 0.0,
                    'landmarks': self.sample_landmarks
                },
                {
                    'frame_idx': 1,
                    'timestamp': 0.033,
                    'landmarks': self.sample_landmarks
                }
            ]
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of JointAngleCalculator."""
        self.assertIsNotNone(self.calculator)
        self.assertTrue(hasattr(self.calculator, 'landmark_indices'))
    
    def test_calculate_angle_between_vectors(self):
        """Test angle calculation between two vectors."""
        from src.feature_engineering.joint_angles import calculate_angle
        
        # Test with known angle (90 degrees)
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        
        angle = calculate_angle(v1, v2)
        self.assertAlmostEqual(angle, 90.0, places=1)
        
        # Test with parallel vectors (0 degrees)
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        
        angle = calculate_angle(v1, v2)
        self.assertAlmostEqual(angle, 0.0, places=1)
    
    def test_landmark_indices_structure(self):
        """Test that landmark indices are properly structured."""
        indices = self.calculator.landmark_indices
        
        self.assertIsInstance(indices, dict)
        self.assertGreater(len(indices), 0)
        
        # Check that indices contain expected joints
        expected_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
        for joint in expected_joints:
            self.assertIn(joint, indices)
            self.assertIsInstance(indices[joint], int)
    
    def test_process_pose_data_structure(self):
        """Test that process_pose_data returns proper DataFrame structure."""
        # Create a test pose data file
        test_file = os.path.join(self.temp_dir, 'test_pose.json')
        with open(test_file, 'w') as f:
            json.dump(self.sample_pose_data, f)
        
        result_df = self.calculator.process_pose_data(test_file)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # Check that expected columns exist
        expected_columns = ['frame_idx', 'timestamp']
        for col in expected_columns:
            self.assertIn(col, result_df.columns)
    
    def test_calculate_vector_function(self):
        """Test the calculate_vector utility function."""
        from src.feature_engineering.joint_angles import calculate_vector
        
        point1 = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        point2 = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        
        vector = calculate_vector(point1, point2)
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), 3)
        np.testing.assert_array_almost_equal(vector, [1.0, 1.0, 1.0])
    
    def test_process_pose_data_with_proper_format(self):
        """Test processing pose data with properly formatted landmarks."""
        # Create properly formatted sample data with dict landmarks
        proper_landmarks = []
        for i in range(33):  # MediaPipe has 33 pose landmarks
            proper_landmarks.append({
                'x': np.random.uniform(0, 1),
                'y': np.random.uniform(0, 1),
                'z': np.random.uniform(0, 1),
                'visibility': np.random.uniform(0.5, 1.0)
            })
        
        proper_pose_data = {
            'frames': [
                {
                    'frame_idx': 0,
                    'timestamp': 0.0,
                    'landmarks': proper_landmarks
                }
            ]
        }
        
        # Create a test pose data file
        test_file = os.path.join(self.temp_dir, 'test_pose_proper.json')
        with open(test_file, 'w') as f:
            json.dump(proper_pose_data, f)
        
        result_df = self.calculator.process_pose_data(test_file)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        
        # Check that expected columns exist
        expected_columns = ['frame_idx', 'timestamp']
        for col in expected_columns:
            self.assertIn(col, result_df.columns)
    
    def test_process_invalid_pose_data(self):
        """Test handling of invalid pose data."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.calculator.process_pose_data('nonexistent_file.json')
        
        # Test with invalid JSON
        invalid_file = os.path.join(self.temp_dir, 'invalid.json')
        with open(invalid_file, 'w') as f:
            f.write('invalid json content')
        
        with self.assertRaises(json.JSONDecodeError):
            self.calculator.process_pose_data(invalid_file)
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction produces consistent results."""
        # Create properly formatted sample data
        proper_landmarks = []
        for i in range(33):
            proper_landmarks.append({
                'x': 0.5,  # Fixed values for consistency
                'y': 0.5,
                'z': 0.5,
                'visibility': 0.8
            })
        
        proper_pose_data = {
            'frames': [
                {
                    'frame_idx': 0,
                    'timestamp': 0.0,
                    'landmarks': proper_landmarks
                }
            ]
        }
        
        # Create test files
        test_file1 = os.path.join(self.temp_dir, 'test_consistency1.json')
        test_file2 = os.path.join(self.temp_dir, 'test_consistency2.json')
        
        with open(test_file1, 'w') as f:
            json.dump(proper_pose_data, f)
        with open(test_file2, 'w') as f:
            json.dump(proper_pose_data, f)
        
        # Calculate features multiple times
        features1 = self.calculator.process_pose_data(test_file1)
        features2 = self.calculator.process_pose_data(test_file2)
        
        # Results should be identical for same input
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_handle_missing_landmarks(self):
        """Test handling of missing or invalid landmarks."""
        # Test with incomplete landmarks (only 20 instead of 33)
        incomplete_landmarks = []
        for i in range(20):
            incomplete_landmarks.append({
                'x': np.random.uniform(0, 1),
                'y': np.random.uniform(0, 1),
                'z': np.random.uniform(0, 1),
                'visibility': np.random.uniform(0.5, 1.0)
            })
        
        incomplete_pose_data = {
            'frames': [
                {
                    'frame_idx': 0,
                    'timestamp': 0.0,
                    'landmarks': incomplete_landmarks
                }
            ]
        }
        
        test_file = os.path.join(self.temp_dir, 'test_incomplete.json')
        with open(test_file, 'w') as f:
            json.dump(incomplete_pose_data, f)
        
        # Should handle gracefully
        result_df = self.calculator.process_pose_data(test_file)
        self.assertIsInstance(result_df, pd.DataFrame)
    
    def test_biomechanical_features(self):
        """Test calculation of cricket-specific biomechanical features."""
        # Create properly formatted sample data
        proper_landmarks = []
        for i in range(33):
            proper_landmarks.append({
                'x': np.random.uniform(0, 1),
                'y': np.random.uniform(0, 1),
                'z': np.random.uniform(0, 1),
                'visibility': np.random.uniform(0.5, 1.0)
            })
        
        proper_pose_data = {
            'frames': [
                {
                    'frame_idx': 0,
                    'timestamp': 0.0,
                    'landmarks': proper_landmarks
                }
            ]
        }
        
        test_file = os.path.join(self.temp_dir, 'test_biomech.json')
        with open(test_file, 'w') as f:
            json.dump(proper_pose_data, f)
        
        result_df = self.calculator.process_pose_data(test_file)
        
        # Check for cricket-specific angles if they exist
        cricket_angles = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle']
        for angle_name in cricket_angles:
            if angle_name in result_df.columns:
                angle_values = result_df[angle_name].dropna()
                if len(angle_values) > 0:
                    for angle_value in angle_values:
                        self.assertIsInstance(angle_value, (int, float))
                        self.assertGreaterEqual(angle_value, 0)
                        self.assertLessEqual(angle_value, 180)


if __name__ == '__main__':
    unittest.main()
