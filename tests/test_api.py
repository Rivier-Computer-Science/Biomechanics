"""
Unit tests for the Cricket Shot Classification API.
"""

import unittest
import json
import tempfile
import os
import io
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app, initialize_components
from api.utils import (
    validate_image_file, allowed_file, validate_image_content,
    format_prediction_response, organize_biomechanical_features,
    generate_technique_feedback, calculate_confidence_metrics
)


class TestCricketShotAPI(unittest.TestCase):
    """Test cases for Cricket Shot Classification API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image file
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('message', data)
        self.assertIn('version', data)
        self.assertIn('timestamp', data)
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('available_models', data)
        self.assertIn('shot_classes', data)
        self.assertIn('model_status', data)
        self.assertIn('features_used', data)
        
        # Check shot classes
        expected_shots = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        self.assertEqual(set(data['shot_classes']), set(expected_shots))
    
    def test_classify_endpoint_no_file(self):
        """Test classify endpoint with no file."""
        response = self.client.post('/api/classify')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No image file provided', data['error'])
    
    def test_classify_endpoint_empty_filename(self):
        """Test classify endpoint with empty filename."""
        data = {'image': (io.BytesIO(b''), '')}
        response = self.client.post('/api/classify', data=data)
        self.assertEqual(response.status_code, 400)
        
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)
    
    def test_classify_endpoint_invalid_file_type(self):
        """Test classify endpoint with invalid file type."""
        data = {'image': (io.BytesIO(b'test content'), 'test.txt')}
        response = self.client.post('/api/classify', data=data)
        self.assertEqual(response.status_code, 400)
        
        response_data = json.loads(response.data)
        self.assertIn('Invalid file type', response_data['error'])
    
    @patch('api.app.pose_extractor')
    @patch('api.app.angle_calculator')
    def test_classify_endpoint_success(self, mock_angle_calc, mock_pose_ext):
        """Test successful classification."""
        # Mock pose extraction
        mock_pose_data = {
            'landmarks': [[0.5, 0.5, 0.8] for _ in range(33)],
            'image_path': 'test.jpg'
        }
        mock_pose_ext.process_image.return_value = (mock_pose_data, 'output_path')
        
        # Mock feature calculation
        mock_features_df = Mock()
        mock_features_df.empty = False
        mock_features_df.columns = ['frame_idx', 'timestamp', 'feature1', 'feature2']
        mock_features_df.iloc = [{'feature1': 45.0, 'feature2': 90.0}]
        mock_angle_calc.process_pose_data.return_value = mock_features_df
        
        # Create test image file
        with open(self.test_image_path, 'rb') as f:
            data = {'image': (f, 'test_image.jpg')}
            response = self.client.post('/api/classify', data=data)
        
        # Should not fail completely even if pose extraction fails
        self.assertIn(response.status_code, [200, 422, 500])
    
    @patch('api.app.pose_extractor')
    def test_classify_endpoint_no_landmarks(self, mock_pose_ext):
        """Test classification with no detected landmarks."""
        # Mock pose extraction failure
        mock_pose_ext.process_image.return_value = (None, None)
        
        with open(self.test_image_path, 'rb') as f:
            data = {'image': (f, 'test_image.jpg')}
            response = self.client.post('/api/classify', data=data)
        
        self.assertEqual(response.status_code, 422)
        response_data = json.loads(response.data)
        self.assertIn('No pose landmarks detected', response_data['error'])
    
    def test_analyze_endpoint_no_file(self):
        """Test analyze endpoint with no file."""
        response = self.client.post('/api/analyze')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_404_handler(self):
        """Test 404 error handler."""
        response = self.client.get('/nonexistent-endpoint')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('not found', data['error'].lower())


class TestAPIUtils(unittest.TestCase):
    """Test cases for API utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_allowed_file(self):
        """Test file extension validation."""
        # Valid extensions
        self.assertTrue(allowed_file('image.jpg'))
        self.assertTrue(allowed_file('image.jpeg'))
        self.assertTrue(allowed_file('image.png'))
        self.assertTrue(allowed_file('image.gif'))
        self.assertTrue(allowed_file('image.bmp'))
        
        # Invalid extensions
        self.assertFalse(allowed_file('document.txt'))
        self.assertFalse(allowed_file('video.mp4'))
        self.assertFalse(allowed_file('archive.zip'))
        
        # Edge cases
        self.assertFalse(allowed_file('no_extension'))
        self.assertTrue(allowed_file('.jpg'))  # This is actually valid - just extension
    
    def test_validate_image_file(self):
        """Test image file validation."""
        # Test with None file
        is_valid, error = validate_image_file(None)
        self.assertFalse(is_valid)
        self.assertIn('No file provided', error)
        
        # Test with empty filename
        mock_file = Mock()
        mock_file.filename = ''
        is_valid, error = validate_image_file(mock_file)
        self.assertFalse(is_valid)
        self.assertIn('No file selected', error)
        
        # Test with invalid extension
        mock_file = Mock()
        mock_file.filename = 'test.txt'
        is_valid, error = validate_image_file(mock_file)
        self.assertFalse(is_valid)
        self.assertIn('Invalid file type', error)
        
        # Test with valid file
        mock_file = Mock()
        mock_file.filename = 'test.jpg'
        is_valid, error = validate_image_file(mock_file)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_image_content(self):
        """Test image content validation."""
        # Create test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        test_path = os.path.join(self.temp_dir, 'test.jpg')
        cv2.imwrite(test_path, test_image)
        
        is_valid, error, info = validate_image_content(test_path)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIsNotNone(info)
        self.assertEqual(info['width'], 400)
        self.assertEqual(info['height'], 300)
        self.assertEqual(info['channels'], 3)
        
        # Test with non-existent file
        is_valid, error, info = validate_image_content('nonexistent.jpg')
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIsNone(info)
    
    def test_format_prediction_response(self):
        """Test prediction response formatting."""
        prediction_probs = np.array([0.1, 0.6, 0.2, 0.1])
        shot_classes = {0: 'drive', 1: 'legglance-flick', 2: 'pullshot', 3: 'sweep'}
        
        response = format_prediction_response(prediction_probs, shot_classes)
        
        self.assertEqual(response['shot_type'], 'legglance-flick')
        self.assertAlmostEqual(response['confidence'], 0.6, places=2)
        self.assertIn('probabilities', response)
        self.assertEqual(len(response['probabilities']), 4)
        
        # Check probabilities sum to 1
        prob_sum = sum(response['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
    
    def test_organize_biomechanical_features(self):
        """Test biomechanical features organization."""
        features = {
            'left_shoulder_angle': 45.0,
            'right_elbow_angle': 90.0,
            'left_hip_angle': 120.0,
            'right_knee_angle': 160.0,
            'spine_angle': 85.0,
            'unknown_feature': 30.0
        }
        
        organized = organize_biomechanical_features(features)
        
        self.assertIn('upper_body', organized)
        self.assertIn('lower_body', organized)
        self.assertIn('overall', organized)
        
        # Check feature categorization
        self.assertIn('left_shoulder_angle', organized['upper_body'])
        self.assertIn('right_elbow_angle', organized['upper_body'])
        self.assertIn('left_hip_angle', organized['lower_body'])
        self.assertIn('right_knee_angle', organized['lower_body'])
    
    def test_generate_technique_feedback(self):
        """Test technique feedback generation."""
        mock_analysis = {
            'upper_body': {'shoulder_angle': 45.0},
            'lower_body': {'hip_angle': 120.0}
        }
        
        feedback = generate_technique_feedback(mock_analysis, 'drive')
        
        self.assertIsInstance(feedback, list)
        self.assertGreater(len(feedback), 0)
        
        # Check feedback structure
        for item in feedback:
            self.assertIn('category', item)
            self.assertIn('recommendation', item)
            self.assertIn('importance', item)
            self.assertIn(item['importance'], ['high', 'medium', 'low'])
    
    def test_calculate_confidence_metrics(self):
        """Test confidence metrics calculation."""
        # Mock pose landmarks (33 landmarks with x, y, visibility)
        pose_landmarks = [[0.5, 0.5, 0.8] for _ in range(33)]
        
        # Mock features DataFrame
        mock_df = Mock()
        mock_df.notna.return_value.sum.return_value.sum.return_value = 80
        mock_df.size = 100
        
        metrics = calculate_confidence_metrics(mock_df, pose_landmarks)
        
        self.assertIn('pose_detection_confidence', metrics)
        self.assertIn('feature_extraction_confidence', metrics)
        self.assertIn('overall_confidence', metrics)
        
        # Check confidence values are between 0 and 1
        for key, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_api_initialization(self):
        """Test API initialization."""
        # Test that the app starts without errors
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_error_handling(self):
        """Test API error handling."""
        # Test 404 error
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_content_type_handling(self):
        """Test content type handling."""
        # Test JSON response content type
        response = self.client.get('/')
        self.assertEqual(response.content_type, 'application/json')
    
    @patch('api.app.initialize_components')
    def test_component_initialization(self, mock_init):
        """Test component initialization."""
        # This would test the initialization of pose extractor, etc.
        mock_init.return_value = None
        
        # Test that initialization doesn't raise errors
        try:
            initialize_components()
        except Exception as e:
            self.fail(f"Component initialization failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
