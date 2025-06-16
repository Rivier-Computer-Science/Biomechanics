#!/usr/bin/env python
"""
Training script for the cricket shot classifier using image pose data.
This script trains a multiclass LSTM model on pose data extracted from cricket shot images.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.multiclass_lstm_model import CricketShotClassifier
from src.feature_engineering.dataset_preparer import CricketShotDatasetPreparer
from src.data_collection.image_pose_extractor import ImagePoseExtractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train cricket shot classifier on image pose data")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/raw/images",
        help="Directory containing the image dataset organized by class"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--pose_dir", 
        type=str, 
        default="data/processed/poses",
        help="Directory to save/load extracted pose data"
    )
    
    parser.add_argument(
        "--features_dir", 
        type=str, 
        default="data/processed/features",
        help="Directory to save/load extracted features"
    )
    
    parser.add_argument(
        "--extract_poses", 
        action="store_true",
        help="Extract poses from images before training"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Save pose visualizations"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="cricket_shot_classifier.pt",
        help="Name of the model file to save"
    )
    
    parser.add_argument(
        "--subset", 
        type=int, 
        default=None,
        help="Use a subset of images per class for initial testing"
    )
    
    return parser.parse_args()


def extract_poses(config, data_dir, pose_dir, visualize=False, subset=None):
    """Extract poses from images."""
    print("Extracting poses from images...")
    
    # Create pose extractor
    extractor = ImagePoseExtractor(
        model_complexity=config.get('pose_estimation', {}).get('model_complexity', 1),
        min_detection_confidence=config.get('pose_estimation', {}).get('min_detection_confidence', 0.5),
        min_tracking_confidence=config.get('pose_estimation', {}).get('min_tracking_confidence', 0.5)
    )
    
    # Process the dataset
    extractor.process_dataset(
        dataset_dir=data_dir,
        output_dir=pose_dir,
        visualize=visualize,
        subset=subset
    )
    
    print(f"Poses extracted and saved to {pose_dir}")


def prepare_dataset(config, pose_dir, features_dir):
    """Prepare the dataset for training."""
    print("Preparing dataset...")
    
    # Create dataset preparer
    preparer = CricketShotDatasetPreparer(config)
    
    # Process pose data
    features_df = preparer.process_pose_files(pose_dir)
    
    # Save combined features
    os.makedirs(features_dir, exist_ok=True)
    features_path = os.path.join(features_dir, "combined_features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"Combined features saved to {features_path}")
    
    # Prepare train/val/test splits
    train_df, val_df, test_df = preparer.prepare_train_test_split(features_df)
    
    # Save splits
    splits_dir = os.path.join(features_dir, "splits")
    preparer.save_splits(train_df, val_df, test_df, splits_dir)
    
    return train_df, val_df, test_df, preparer.class_mapping


def train_model(config, train_df, val_df, output_dir, model_name, class_mapping):
    """Train the classifier model."""
    print("Training model...")
    
    # Create classifier
    classifier = CricketShotClassifier(config_path=None)
    classifier.config = config  # Use the loaded config
    
    # Set class mapping
    classifier.class_mapping = class_mapping
    
    # Prepare sequences
    X_train, y_train = classifier.prepare_sequences(train_df)
    X_val, y_val = classifier.prepare_sequences(val_df)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Train the model
    history = classifier.train(X_train, y_train, X_val, y_val)
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    
    # Save input size in metadata
    metadata = {
        'input_size': X_train.shape[2],
        'class_mapping': class_mapping
    }
    
    classifier.save_model(model_path, metadata)
    
    return classifier, history


def evaluate_model(classifier, test_df, output_dir):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Prepare test data
    X_test, y_test = classifier.prepare_sequences(test_df)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    # Print classification report
    print("\nClassification Report:")
    report = metrics['classification_report']
    for class_name, values in report.items():
        if isinstance(values, dict):
            print(f"Class: {class_name}")
            for metric, value in values.items():
                print(f"  {metric}: {value:.4f}")
    
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = metrics['confusion_matrix']
    
    # Get class names if available
    if classifier.label_encoder is not None:
        class_names = classifier.label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(conf_matrix.shape[0])]
    
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    
    # Convert numpy values to Python native types for JSON serialization
    json_report = {}
    for k, v in report.items():
        if isinstance(v, dict):
            json_report[k] = {m: float(val) for m, val in v.items()}
        else:
            json_report[k] = float(v)
    
    with open(metrics_path, 'w') as f:
        json.dump({'classification_report': json_report}, f, indent=2)
    
    print(f"Evaluation metrics saved to {metrics_path}")
    
    return metrics


def plot_training_history(history, output_dir):
    """Plot and save training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pose_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    
    # Extract poses if requested
    if args.extract_poses:
        extract_poses(config, args.data_dir, args.pose_dir, args.visualize, args.subset)
    
    # Prepare dataset
    train_df, val_df, test_df, class_mapping = prepare_dataset(config, args.pose_dir, args.features_dir)
    
    # Train model
    classifier, history = train_model(config, train_df, val_df, args.output_dir, args.model_name, class_mapping)
    
    # Evaluate model
    metrics = evaluate_model(classifier, test_df, args.output_dir)
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    print("\nTraining and evaluation complete!")
    print(f"Model saved to {os.path.join(args.output_dir, args.model_name)}")


if __name__ == "__main__":
    main()