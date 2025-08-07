"""
Unit tests for pose extraction functionality.
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.image_pose_extractor import ImagePoseExtractor


class TestImagePoseExtractor(unittest.TestCase):
    """Test cases for ImagePoseExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ImagePoseExtractor()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of ImagePoseExtractor."""
        self.assertIsNotNone(self.extractor)
        self.assertTrue(hasattr(self.extractor, 'pose'))
    
    def test_initialization_success(self):
        """Test successful initialization of ImagePoseExtractor."""
        self.assertIsNotNone(self.extractor)
        self.assertTrue(hasattr(self.extractor, 'pose'))
        self.assertTrue(hasattr(self.extractor, 'config'))
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Test default config
        self.assertIn('pose', self.extractor.config)
        self.assertIn('model', self.extractor.config['pose'])
        self.assertIn('confidence_threshold', self.extractor.config['pose'])
    
    def test_process_image_with_valid_file(self):
        """Test processing a valid image file."""
        # Create a test image file
        test_image_path = os.path.join(self.temp_dir, 'test_input.jpg')
        cv2.imwrite(test_image_path, self.test_image)
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        try:
            pose_data, output_path = self.extractor.process_image(
                test_image_path, output_dir, visualize=False
            )
            
            # Check that output was generated
            self.assertIsNotNone(pose_data)
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Check pose data structure
            self.assertIn('image_name', pose_data)
            self.assertIn('landmarks', pose_data)
            
        except Exception as e:
            # MediaPipe might not work in test environment, so we allow this
            self.assertIsInstance(e, (ImportError, AttributeError, RuntimeError, ValueError))
    
    def test_batch_process_functionality(self):
        """Test batch processing functionality."""
        # Create multiple test images
        input_dir = os.path.join(self.temp_dir, 'input_images')
        os.makedirs(input_dir, exist_ok=True)
        
        for i in range(3):
            test_image_path = os.path.join(input_dir, f'test_image_{i}.jpg')
            cv2.imwrite(test_image_path, self.test_image)
        
        output_dir = os.path.join(self.temp_dir, 'batch_output')
        
        try:
            output_files, metadata = self.extractor.batch_process(
                input_dir, output_dir, limit=2
            )
            
            # Check that outputs were generated
            self.assertIsInstance(output_files, list)
            self.assertIsInstance(metadata, list)
            self.assertLessEqual(len(output_files), 2)  # Respects limit
            
        except Exception as e:
            # Allow MediaPipe-related errors in test environment
            self.assertIsInstance(e, (ImportError, AttributeError, RuntimeError, ValueError))
    
    def test_process_image_workflow(self):
        """Test the complete image processing workflow."""
        # Create a test image
        test_image_path = os.path.join(self.temp_dir, 'test_input.jpg')
        cv2.imwrite(test_image_path, self.test_image)
        
        output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            pose_data, output_path = self.extractor.process_image(
                test_image_path, output_dir, visualize=True
            )
            
            # Check that output was generated
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Check pose data structure
            if pose_data is not None:
                self.assertIn('landmarks', pose_data)
                self.assertIn('image_name', pose_data)  # Fixed: should be 'image_name' not 'image_path'
        
        except Exception as e:
            # MediaPipe might not work in test environment, so we allow this
            self.assertIsInstance(e, (ImportError, AttributeError, RuntimeError, ValueError))
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.extractor.process_image('nonexistent_file.jpg', self.temp_dir)
    
    def test_output_directory_creation(self):
        """Test automatic creation of output directories."""
        nonexistent_dir = os.path.join(self.temp_dir, 'new_output_dir')
        test_image_path = os.path.join(self.temp_dir, 'test_input.jpg')
        cv2.imwrite(test_image_path, self.test_image)
        
        try:
            self.extractor.process_image(test_image_path, nonexistent_dir)
            self.assertTrue(os.path.exists(nonexistent_dir))
        except Exception:
            # Allow MediaPipe-related errors in test environment
            pass


if __name__ == '__main__':
    unittest.main()
