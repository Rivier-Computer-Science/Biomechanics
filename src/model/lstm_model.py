"""
LSTM-based neural network model for cricket straight drive analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import json
import os


class StraightDriveLSTM(nn.Module):
    """
    LSTM neural network for cricket straight drive technique classification.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability
        """
        super(StraightDriveLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))
        
        # We use the last time step output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class StraightDriveClassifier:
    """Wrapper class for training and evaluating the LSTM model."""
    
    def __init__(self, config_path=None):
        """
        Initialize the classifier with configurations.
        
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
                'model': {
                    'type': 'lstm',
                    'sequence_length': 60,
                    'hidden_units': 128,
                    'dropout': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def build_model(self, input_size, num_classes):
        """
        Build the LSTM model.
        
        Args:
            input_size (int): Number of input features
            num_classes (int): Number of output classes
            
        Returns:
            StraightDriveLSTM: The initialized model
        """
        model_config = self.config['model']
        
        self.model = StraightDriveLSTM(
            input_size=input_size,
            hidden_size=model_config.get('hidden_units', 128),
            num_layers=model_config.get('num_layers', 2),
            num_classes=num_classes,
            dropout_rate=model_config.get('dropout', 0.3)
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config.get('learning_rate', 0.001)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        return self.model
    
    def prepare_sequences(self, features_df, label_col=None, seq_length=None):
        """
        Prepare sequential data for the LSTM model.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features
            label_col (str): Column name for labels
            seq_length (int): Length of sequences
            
        Returns:
            tuple: (sequences, labels if label_col provided)
        """
        if seq_length is None:
            seq_length = self.config['model'].get('sequence_length', 60)
        
        # Get feature columns (exclude metadata columns)
        exclude_cols = ['frame_idx', 'timestamp']
        if label_col:
            exclude_cols.append(label_col)
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Create sequences
        sequences = []
        labels = [] if label_col else None
        
        for i in range(0, len(features_df) - seq_length + 1):
            seq = features_df.iloc[i:i+seq_length][feature_cols].values
            sequences.append(seq)
            
            if label_col:
                # Use the label of the last frame in the sequence
                label = features_df.iloc[i+seq_length-1][label_col]
                labels.append(label)
        
        # Convert to numpy arrays
        sequences = np.array(sequences)
        if label_col:
            labels = np.array(labels)
            return sequences, labels
        else:
            return sequences
    
    def train(self, train_features, train_labels, val_features=None, val_labels=None):
        """
        Train the LSTM model.
        
        Args:
            train_features (np.array): Training features
            train_labels (np.array): Training labels
            val_features (np.array): Validation features
            val_labels (np.array): Validation labels
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            input_size = train_features.shape[2]  # Number of features per time step
            num_classes = len(np.unique(train_labels))
            self.build_model(input_size, num_classes)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(train_features).to(self.device)
        y_train = torch.LongTensor(train_labels).to(self.device)
        
        if val_features is not None and val_labels is not None:
            X_val = torch.FloatTensor(val_features).to(self.device)
            y_val = torch.LongTensor(val_labels).to(self.device)
            has_validation = True
        else:
            has_validation = False
        
        # Training parameters
        batch_size = self.config['model'].get('batch_size', 32)
        epochs = self.config['model'].get('epochs', 100)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': []
        }
        
        if has_validation:
            history.update({
                'val_loss': [],
                'val_acc': []
            })
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track loss and accuracy
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / (len(X_train) / batch_size)
            train_accuracy = train_correct / len(X_train)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation
            if has_validation:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                
                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        batch_X = X_val[i:i+batch_size]
                        batch_y = y_val[i:i+batch_size]
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / (len(X_val) / batch_size)
                val_accuracy = val_correct / len(X_val)
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        
        return history
    
    def evaluate(self, test_features, test_labels):
        """
        Evaluate the model on test data.
        
        Args:
            test_features (np.array): Test features
            test_labels (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert to PyTorch tensors
        X_test = torch.FloatTensor(test_features).to(self.device)
        y_test = torch.LongTensor(test_labels).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy for metrics calculation
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, features):
        """
        Make predictions with the trained model.
        
        Args:
            features (np.array): Features to predict
            
        Returns:
            np.array: Predicted classes
            np.array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(features).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy
        predicted = predicted.cpu().numpy()
        probs = probs.cpu().numpy()
        
        return predicted, probs
    
    def save_model(self, model_path, metadata=None):
        """
        Save the trained model and metadata.
        
        Args:
            model_path (str): Path to save the model
            metadata (dict): Additional metadata to save
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = Path(model_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path, input_size, num_classes):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            input_size (int): Number of input features
            num_classes (int): Number of output classes
            
        Returns:
            StraightDriveLSTM: The loaded model
        """
        # Build model architecture
        self.build_model(input_size, num_classes)
        
        # Load state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Set to evaluation mode
        self.model.eval()
        
        return self.model


if __name__ == "__main__":
    # Example usage
    classifier = StraightDriveClassifier("../../configs/config.yaml")
    
    # Training would look like this:
    # X_train, y_train = prepare_data(...)
    # X_val, y_val = prepare_data(...)
    # history = classifier.train(X_train, y_train, X_val, y_val)
    
    # Saving the model:
    # classifier.save_model("../../models/straight_drive_classifier.pt")
