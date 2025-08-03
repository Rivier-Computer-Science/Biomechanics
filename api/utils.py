"""
Utility functions for the Cricket Shot Classification API.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from werkzeug.utils import secure_filename

from .config import UPLOAD_CONFIG, MODEL_CONFIG


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_image_file(file):
    """
    Validate uploaded image file.
    
    Args:
        file: Flask file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        allowed_exts = ', '.join(UPLOAD_CONFIG['allowed_extensions'])
        return False, f"Invalid file type. Allowed extensions: {allowed_exts}"
    
    return True, None


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in UPLOAD_CONFIG['allowed_extensions']


def save_uploaded_file(file, temp_dir):
    """
    Save uploaded file to temporary directory.
    
    Args:
        file: Flask file object
        temp_dir: Temporary directory path
        
    Returns:
        str: Path to saved file
    """
    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    file_path = os.path.join(temp_dir, unique_filename)
    file.save(file_path)
    return file_path


def validate_image_content(image_path):
    """
    Validate that the file is actually a valid image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (is_valid, error_message, image_info)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, "Invalid image file or corrupted data", None
        
        height, width, channels = image.shape
        
        # Check minimum dimensions
        if height < 100 or width < 100:
            return False, "Image too small. Minimum size: 100x100 pixels", None
        
        # Check maximum dimensions
        if height > 4000 or width > 4000:
            return False, "Image too large. Maximum size: 4000x4000 pixels", None
        
        image_info = {
            'width': width,
            'height': height,
            'channels': channels,
            'size_bytes': os.path.getsize(image_path)
        }
        
        return True, None, image_info
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}", None


def format_prediction_response(prediction_probs, shot_classes):
    """
    Format model prediction into API response format.
    
    Args:
        prediction_probs: Array of prediction probabilities
        shot_classes: Dictionary mapping class indices to names
        
    Returns:
        dict: Formatted prediction response
    """
    predicted_class_idx = np.argmax(prediction_probs)
    predicted_class = shot_classes[predicted_class_idx]
    confidence = float(prediction_probs[predicted_class_idx])
    
    probabilities = {
        shot_name: float(prob) 
        for idx, (shot_name, prob) in enumerate(zip(shot_classes.values(), prediction_probs))
    }
    
    return {
        'shot_type': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }


def organize_biomechanical_features(features_dict):
    """
    Organize extracted features by body part categories.
    
    Args:
        features_dict: Dictionary of feature names and values
        
    Returns:
        dict: Features organized by body part
    """
    categories = {
        'upper_body': {
            'keywords': ['shoulder', 'elbow', 'wrist', 'arm', 'chest'],
            'features': {}
        },
        'lower_body': {
            'keywords': ['hip', 'knee', 'ankle', 'leg', 'foot'],
            'features': {}
        },
        'core': {
            'keywords': ['spine', 'torso', 'back', 'core'],
            'features': {}
        },
        'overall': {
            'keywords': ['posture', 'balance', 'center'],
            'features': {}
        }
    }
    
    # Categorize features
    for feature_name, value in features_dict.items():
        categorized = False
        
        for category, info in categories.items():
            if any(keyword in feature_name.lower() for keyword in info['keywords']):
                categories[category]['features'][feature_name] = value
                categorized = True
                break
        
        # If not categorized, put in overall
        if not categorized:
            categories['overall']['features'][feature_name] = value
    
    # Remove empty categories and return only features
    result = {}
    for category, info in categories.items():
        if info['features']:
            result[category] = info['features']
    
    return result


def generate_technique_feedback(biomechanical_analysis, predicted_shot):
    """
    Generate technique feedback based on biomechanical analysis.
    
    Args:
        biomechanical_analysis: Dictionary of organized features
        predicted_shot: Predicted cricket shot type
        
    Returns:
        list: List of feedback recommendations
    """
    feedback = []
    
    # Shot-specific feedback
    shot_feedback = {
        'drive': [
            {
                'category': 'Stance',
                'recommendation': 'Keep front foot close to the pitch of the ball',
                'importance': 'high'
            },
            {
                'category': 'Bat Position',
                'recommendation': 'Maintain straight bat with full face presentation',
                'importance': 'high'
            }
        ],
        'pullshot': [
            {
                'category': 'Body Position',
                'recommendation': 'Transfer weight to back foot and rotate hips',
                'importance': 'high'
            },
            {
                'category': 'Timing',
                'recommendation': 'Play the shot in front of square leg',
                'importance': 'medium'
            }
        ],
        'sweep': [
            {
                'category': 'Footwork',
                'recommendation': 'Get front pad close to the pitch of the ball',
                'importance': 'high'
            },
            {
                'category': 'Bat Angle',
                'recommendation': 'Keep bat horizontal and sweep along the ground',
                'importance': 'high'
            }
        ],
        'legglance-flick': [
            {
                'category': 'Wrist Position',
                'recommendation': 'Use strong wrist action to guide the ball',
                'importance': 'high'
            },
            {
                'category': 'Balance',
                'recommendation': 'Maintain balance while moving across the stumps',
                'importance': 'medium'
            }
        ]
    }
    
    # Add shot-specific feedback
    if predicted_shot in shot_feedback:
        feedback.extend(shot_feedback[predicted_shot])
    
    # Add general feedback based on analysis
    feedback.extend([
        {
            'category': 'General',
            'recommendation': 'Keep head still and eyes level throughout the shot',
            'importance': 'high'
        },
        {
            'category': 'Follow Through',
            'recommendation': 'Complete the shot with proper follow-through',
            'importance': 'medium'
        }
    ])
    
    return feedback


def calculate_confidence_metrics(features_df, pose_landmarks):
    """
    Calculate confidence metrics for the analysis.
    
    Args:
        features_df: DataFrame with extracted features
        pose_landmarks: List of pose landmarks
        
    Returns:
        dict: Confidence metrics
    """
    metrics = {
        'pose_detection_confidence': 0.0,
        'feature_extraction_confidence': 0.0,
        'overall_confidence': 0.0
    }
    
    # Calculate pose detection confidence
    if pose_landmarks:
        visible_landmarks = sum(1 for landmark in pose_landmarks if landmark[2] > 0.5)  # visibility threshold
        metrics['pose_detection_confidence'] = visible_landmarks / len(pose_landmarks)
    
    # Calculate feature extraction confidence
    if features_df is not None and not features_df.empty:
        valid_features = features_df.notna().sum().sum()
        total_features = features_df.size
        metrics['feature_extraction_confidence'] = valid_features / total_features if total_features > 0 else 0.0
    
    # Calculate overall confidence
    metrics['overall_confidence'] = (
        metrics['pose_detection_confidence'] * 0.6 + 
        metrics['feature_extraction_confidence'] * 0.4
    )
    
    return metrics


def create_error_response(error_message, error_code=None, details=None):
    """
    Create standardized error response.
    
    Args:
        error_message: Main error message
        error_code: Optional error code
        details: Optional additional details
        
    Returns:
        dict: Standardized error response
    """
    response = {
        'success': False,
        'error': error_message,
        'timestamp': datetime.now().isoformat()
    }
    
    if error_code:
        response['error_code'] = error_code
    
    if details:
        response['details'] = details
    
    return response


def create_success_response(data, message=None):
    """
    Create standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        dict: Standardized success response
    """
    response = {
        'success': True,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    
    return response
