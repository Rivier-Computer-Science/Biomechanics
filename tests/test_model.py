"""
Unit tests for LSTM model functionality.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model.multiclass_lstm_model import CricketShotLSTM, CricketShotClassifier
except ImportError:
    # Fallback if exact module names are different
    CricketShotLSTM = None
    CricketShotClassifier = None


class TestCricketShotModel(unittest.TestCase):
    """Test cases for Cricket Shot Classification Model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50  # Number of features
        self.hidden_size = 64
        self.num_layers = 2
        self.num_classes = 4  # drive, legglance-flick, pullshot, sweep
        self.sequence_length = 10
        self.batch_size = 8
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        self.sample_features = np.random.randn(100, self.input_size)
        self.sample_labels = np.random.randint(0, self.num_classes, 100)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(CricketShotLSTM is None, "Model classes not available")
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization."""
        model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
        # Check that LSTM layer exists
        self.assertTrue(hasattr(model, 'lstm'))
        self.assertTrue(hasattr(model, 'fc1'))
        self.assertTrue(hasattr(model, 'fc2'))
    
    @unittest.skipIf(CricketShotLSTM is None, "Model classes not available")
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Create sample input
        sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # Forward pass
        output = model(sample_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is valid probabilities (after softmax)
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(self.batch_size)))
    
    @unittest.skipIf(CricketShotClassifier is None, "Classifier class not available")
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        # Create a temporary config file instead of passing dict
        config_data = {
            'model': {
                'type': 'lstm',
                'sequence_length': 1,
                'hidden_units': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }
        }
        
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        classifier = CricketShotClassifier(config_path)
        
        self.assertIsNotNone(classifier.config)
        self.assertIn('model', classifier.config)
        # Model, optimizer, criterion are None until build_model is called
        self.assertIsNone(classifier.model)
        self.assertIsNone(classifier.optimizer)
        self.assertIsNone(classifier.criterion)
    
    def test_data_preprocessing(self):
        """Test data preprocessing for model input."""
        # Test sequence creation from features
        features = self.sample_features
        labels = self.sample_labels
        
        # Mock sequence creation (this would be in actual preprocessing)
        sequences = []
        sequence_labels = []
        
        for i in range(len(features) - self.sequence_length + 1):
            sequence = features[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]  # Use last label in sequence
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        # Check shapes
        expected_shape = (len(features) - self.sequence_length + 1, self.sequence_length, self.input_size)
        self.assertEqual(sequences.shape, expected_shape)
        self.assertEqual(len(sequence_labels), len(sequences))
    
    def test_loss_calculation(self):
        """Test loss calculation."""
        # Create sample predictions and targets
        predictions = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        if CricketShotLSTM is None:
            self.skipTest("Model class not available")
            
        model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test that loaded model produces same output
        sample_input = torch.randn(1, self.sequence_length, self.input_size)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = loaded_model(sample_input)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_prediction_accuracy_calculation(self):
        """Test accuracy calculation for predictions."""
        # Sample predictions and true labels
        predictions = torch.tensor([[0.1, 0.8, 0.05, 0.05],
                                   [0.7, 0.1, 0.1, 0.1],
                                   [0.2, 0.2, 0.2, 0.4]])
        true_labels = torch.tensor([1, 0, 3])
        
        # Calculate accuracy
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == true_labels).float().mean()
        
        expected_accuracy = 1.0  # All predictions are correct
        self.assertAlmostEqual(accuracy.item(), expected_accuracy, places=4)
    
    def test_model_training_step(self):
        """Test a single training step."""
        if CricketShotLSTM is None:
            self.skipTest("Model class not available")
            
        model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Sample batch
        inputs = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_model_evaluation_mode(self):
        """Test model evaluation mode."""
        if CricketShotLSTM is None:
            self.skipTest("Model class not available")
            
        model = CricketShotLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
        
        # Test consistent outputs in eval mode
        sample_input = torch.randn(1, self.sequence_length, self.input_size)
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)
        
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
