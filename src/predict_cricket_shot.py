#!/usr/bin/env python
"""
Prediction script for the cricket shot classifier.
This script uses a trained model to predict cricket shots from images.
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
import cv2
from glob import glob

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.multiclass_lstm_model import CricketShotClassifier
from src.data_collection.image_pose_extractor import ImagePoseExtractor
from src.feature_engineering.joint_angles import JointAngleCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict cricket shots from images")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the trained model file"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input image or directory of images"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize pose and prediction on the image"
    )
    
    return parser.parse_args()


def load_model(model_path, config):
    """Load the trained classifier model."""
    print(f"Loading model from {model_path}...")
    
    # Load metadata to get input size
    metadata_path = Path(model_path).with_suffix('.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_size = metadata.get('input_size')
    if input_size is None:
        raise ValueError("Input size not found in model metadata")
    
    # Get number of classes from metadata
    if 'classes' in metadata:
        num_classes = len(metadata['classes'])
    elif 'class_mapping' in metadata:
        num_classes = len(metadata['class_mapping'])
    else:
        raise ValueError("Class information not found in model metadata")
    
    # Create and load classifier
    classifier = CricketShotClassifier(config_path=None)
    classifier.config = config  # Use the loaded config
    classifier.load_model(model_path, input_size=input_size, num_classes=num_classes)
    
    return classifier


def process_image(image_path, pose_extractor, angle_calculator):
    """Process a single image to extract features."""
    # Extract pose
    pose_data = pose_extractor.process_image(image_path)
    
    if not pose_data or 'landmarks' not in pose_data:
        print(f"No pose detected in {image_path}")
        return None
    
    # Calculate joint angles
    features_df = angle_calculator.process_pose_data(pose_data)
    
    # Drop unnecessary columns for prediction
    if 'frame_idx' in features_df.columns:
        features_df = features_df.drop('frame_idx', axis=1)
    if 'timestamp' in features_df.columns:
        features_df = features_df.drop('timestamp', axis=1)
    
    return features_df


def predict_shot(image_path, classifier, pose_extractor, angle_calculator):
    """Predict the cricket shot for a single image."""
    # Process image
    features_df = process_image(image_path, pose_extractor, angle_calculator)
    
    if features_df is None or features_df.empty:
        return None, None
    
    # Prepare for prediction (add sequence dimension)
    features = features_df.values.reshape(1, 1, -1)  # (1, seq_len=1, features)
    
    # Predict
    predicted_class, probabilities = classifier.predict(features)
    
    return predicted_class[0], probabilities[0]


def visualize_prediction(image_path, predicted_class, probabilities, output_path, pose_extractor):
    """Visualize the prediction on the image."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return
    
    # Draw pose landmarks
    image_with_pose = pose_extractor.draw_landmarks_on_image(image_path)
    if image_with_pose is None:
        image_with_pose = image.copy()
    
    # Add prediction text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)  # Green
    
    # Format prediction text
    text = f"Predicted: {predicted_class}"
    cv2.putText(image_with_pose, text, (10, 30), font, font_scale, color, thickness)
    
    # Add top 3 probabilities
    if probabilities is not None:
        # Get class names
        class_names = classifier.label_encoder.classes_
        
        # Get top 3 indices
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        for i, idx in enumerate(top_indices):
            prob_text = f"{class_names[idx]}: {probabilities[idx]:.2f}"
            cv2.putText(image_with_pose, prob_text, (10, 70 + i*40), font, font_scale, color, thickness)
    
    # Save the visualization
    cv2.imwrite(output_path, image_with_pose)
    print(f"Visualization saved to {output_path}")


def process_directory(input_dir, output_dir, classifier, pose_extractor, angle_calculator, visualize=False):
    """Process all images in a directory."""
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    results = []
    
    for image_path in image_files:
        print(f"Processing {image_path}...")
        
        try:
            # Predict
            predicted_class, probabilities = predict_shot(
                image_path, classifier, pose_extractor, angle_calculator
            )
            
            if predicted_class is None:
                print(f"Could not predict for {image_path}")
                continue
            
            # Create result dictionary
            result = {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": float(np.max(probabilities)) if probabilities is not None else None
            }
            
            results.append(result)
            
            print(f"Predicted: {predicted_class} with confidence {result['confidence']:.2f}")
            
            # Visualize if requested
            if visualize:
                output_filename = f"{Path(image_path).stem}_prediction{Path(image_path).suffix}"
                output_path = os.path.join(output_dir, output_filename)
                visualize_prediction(image_path, predicted_class, probabilities, output_path, pose_extractor)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, "prediction_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Generate summary
    class_counts = {}
    for result in results:
        cls = result["predicted_class"]
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    print("\nPrediction Summary:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images ({count/len(results)*100:.1f}%)")


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    classifier = load_model(args.model, config)
    
    # Create pose extractor
    pose_extractor = ImagePoseExtractor(
        model_complexity=config.get('pose_estimation', {}).get('model_complexity', 1),
        min_detection_confidence=config.get('pose_estimation', {}).get('min_detection_confidence', 0.5),
        min_tracking_confidence=config.get('pose_estimation', {}).get('min_tracking_confidence', 0.5)
    )
    
    # Create joint angle calculator
    angle_calculator = JointAngleCalculator(config)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single image
        print(f"Processing single image: {args.input}")
        predicted_class, probabilities = predict_shot(
            args.input, classifier, pose_extractor, angle_calculator
        )
        
        if predicted_class is not None:
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {np.max(probabilities):.2f}")
            
            # Visualize if requested
            if args.visualize:
                output_filename = f"{Path(args.input).stem}_prediction{Path(args.input).suffix}"
                output_path = os.path.join(args.output_dir, output_filename)
                visualize_prediction(args.input, predicted_class, probabilities, output_path, pose_extractor)
        else:
            print(f"Could not predict for {args.input}")
    
    elif os.path.isdir(args.input):
        # Process directory
        print(f"Processing directory: {args.input}")
        process_directory(
            args.input, args.output_dir, classifier, pose_extractor, angle_calculator, args.visualize
        )
    
    else:
        print(f"Input path {args.input} does not exist")


if __name__ == "__main__":
    main()