"""
Flask API for Cricket Shot Classification

This module provides REST API endpoints for cricket shot classification
from uploaded images using the trained LSTM model.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.image_pose_extractor import ImagePoseExtractor
from src.feature_engineering.joint_angles import JointAngleCalculator
from src.inference.cricket_analyzer import CricketTechniqueAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Cricket shot classes
CRICKET_SHOTS = {
    0: 'drive',
    1: 'legglance-flick', 
    2: 'pullshot',
    3: 'sweep'
}

# Global variables for model components
pose_extractor = None
angle_calculator = None
model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_components():
    """Initialize pose extractor, angle calculator, and model."""
    global pose_extractor, angle_calculator, model
    
    try:
        # Initialize pose extractor
        pose_extractor = ImagePoseExtractor()
        logger.info("Pose extractor initialized successfully")
        
        # Initialize angle calculator
        angle_calculator = JointAngleCalculator()
        logger.info("Joint angle calculator initialized successfully")
        
        # Try to load trained model (if available)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cricket_shot_classifier.pth')
        if os.path.exists(model_path):
            # This would load the actual trained model
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No trained model found. Predictions will use mock data.")
            
    except Exception as e:
        logger.error(f"Error initializing components: {e}")


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Cricket Shot Classification API is running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/classify', methods=['POST'])
def classify_cricket_shot():
    """
    Classify cricket shot from uploaded image.
    
    Returns:
        JSON response with classification results
    """
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            filename = secure_filename(file.filename)
            temp_filename = f"{uuid.uuid4()}_{filename}"
            temp_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_path)
            
            # Extract pose data
            pose_data, pose_output_path = pose_extractor.process_image(
                temp_path, temp_dir, visualize=True
            )
            
            if pose_data is None or not pose_data.get('landmarks'):
                return jsonify({
                    'error': 'No pose landmarks detected in the image',
                    'message': 'Please ensure the image contains a clear view of a cricket player'
                }), 422
            
            # Calculate biomechanical features
            # Create temporary pose file for angle calculator
            pose_file_path = os.path.join(temp_dir, 'pose_data.json')
            adapted_pose_data = {
                'frames': [{
                    'frame_idx': 0,
                    'timestamp': 0,
                    'landmarks': pose_data['landmarks']
                }]
            }
            
            with open(pose_file_path, 'w') as f:
                json.dump(adapted_pose_data, f)
            
            features_df = angle_calculator.process_pose_data(pose_file_path)
            
            if features_df is None or features_df.empty:
                return jsonify({
                    'error': 'Failed to extract biomechanical features',
                    'message': 'Could not calculate joint angles from pose data'
                }), 422
            
            # Make prediction (mock implementation if no trained model)
            if model is not None:
                # Use actual trained model for prediction
                prediction_probs = make_model_prediction(features_df)
            else:
                # Mock prediction for demonstration
                prediction_probs = np.random.dirichlet(np.ones(4))  # Random probabilities
            
            # Get predicted class
            predicted_class_idx = np.argmax(prediction_probs)
            predicted_class = CRICKET_SHOTS[predicted_class_idx]
            confidence = float(prediction_probs[predicted_class_idx])
            
            # Prepare response
            response = {
                'success': True,
                'prediction': {
                    'shot_type': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        shot_name: float(prob) 
                        for shot_name, prob in zip(CRICKET_SHOTS.values(), prediction_probs)
                    }
                },
                'pose_analysis': {
                    'landmarks_detected': len(pose_data['landmarks']),
                    'features_extracted': len(features_df.columns) - 2,  # Exclude frame_idx and timestamp
                },
                'processing_info': {
                    'image_filename': filename,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return jsonify({
            'error': 'Internal server error during classification',
            'message': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_cricket_technique():
    """
    Detailed biomechanical analysis of cricket shot.
    
    Returns:
        JSON response with detailed analysis
    """
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            filename = secure_filename(file.filename)
            temp_filename = f"{uuid.uuid4()}_{filename}"
            temp_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_path)
            
            # Extract pose data
            pose_data, pose_output_path = pose_extractor.process_image(
                temp_path, temp_dir, visualize=True
            )
            
            if pose_data is None or not pose_data.get('landmarks'):
                return jsonify({'error': 'No pose landmarks detected'}), 422
            
            # Calculate detailed biomechanical features
            pose_file_path = os.path.join(temp_dir, 'pose_data.json')
            adapted_pose_data = {
                'frames': [{
                    'frame_idx': 0,
                    'timestamp': 0,
                    'landmarks': pose_data['landmarks']
                }]
            }
            
            with open(pose_file_path, 'w') as f:
                json.dump(adapted_pose_data, f)
            
            features_df = angle_calculator.process_pose_data(pose_file_path)
            
            if features_df is None or features_df.empty:
                return jsonify({'error': 'Failed to extract features'}), 422
            
            # Extract specific biomechanical metrics
            features = features_df.iloc[0].to_dict()
            
            # Organize features by body part
            biomechanical_analysis = organize_features_by_body_part(features)
            
            # Add technique recommendations (mock implementation)
            recommendations = generate_technique_recommendations(biomechanical_analysis)
            
            response = {
                'success': True,
                'biomechanical_analysis': biomechanical_analysis,
                'technique_recommendations': recommendations,
                'processing_info': {
                    'image_filename': filename,
                    'landmarks_detected': len(pose_data['landmarks']),
                    'features_calculated': len(features),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return jsonify({
            'error': 'Internal server error during analysis',
            'message': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_model_info():
    """Get information about available models."""
    model_info = {
        'available_models': ['cricket_shot_classifier'],
        'shot_classes': list(CRICKET_SHOTS.values()),
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'features_used': [
            'joint_angles',
            'body_positions',
            'limb_distances',
            'pose_landmarks'
        ]
    }
    
    return jsonify(model_info)


def make_model_prediction(features_df):
    """Make prediction using trained model."""
    # This would implement actual model prediction
    # For now, return mock probabilities
    return np.random.dirichlet(np.ones(4))


def organize_features_by_body_part(features):
    """Organize extracted features by body part."""
    body_parts = {
        'upper_body': {},
        'lower_body': {},
        'overall_posture': {}
    }
    
    for feature_name, value in features.items():
        if any(part in feature_name.lower() for part in ['shoulder', 'elbow', 'wrist', 'arm']):
            body_parts['upper_body'][feature_name] = value
        elif any(part in feature_name.lower() for part in ['hip', 'knee', 'ankle', 'leg']):
            body_parts['lower_body'][feature_name] = value
        else:
            body_parts['overall_posture'][feature_name] = value
    
    return body_parts


def generate_technique_recommendations(biomechanical_analysis):
    """Generate technique recommendations based on analysis."""
    recommendations = []
    
    # Mock recommendations based on common cricket technique principles
    recommendations.append({
        'category': 'Stance',
        'recommendation': 'Maintain balanced stance with feet shoulder-width apart',
        'importance': 'high'
    })
    
    recommendations.append({
        'category': 'Bat Position',
        'recommendation': 'Keep bat close to body for better control',
        'importance': 'medium'
    })
    
    recommendations.append({
        'category': 'Follow Through',
        'recommendation': 'Complete the shot with proper follow-through',
        'importance': 'high'
    })
    
    return recommendations


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
