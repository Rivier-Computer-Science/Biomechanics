"""
Configuration settings for the Cricket Shot Classification API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_content_length': 16 * 1024 * 1024,  # 16MB
}

# File upload settings
UPLOAD_CONFIG = {
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'max_file_size': 16 * 1024 * 1024,  # 16MB
    'temp_dir': os.path.join(BASE_DIR, 'temp'),
}

# Model configuration
MODEL_CONFIG = {
    'model_path': os.path.join(BASE_DIR, 'models', 'cricket_shot_classifier.pth'),
    'config_path': os.path.join(BASE_DIR, 'configs', 'config.yaml'),
    'shot_classes': {
        0: 'drive',
        1: 'legglance-flick',
        2: 'pullshot',
        3: 'sweep'
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(BASE_DIR, 'logs', 'api.log')
}

# Processing configuration
PROCESSING_CONFIG = {
    'pose_confidence_threshold': 0.5,
    'min_landmarks_required': 20,
    'enable_visualization': True,
    'timeout_seconds': 30
}
