# Cricket Shot Classification API

A REST API for classifying cricket shots from uploaded images using pose estimation and machine learning.

## Features

- **Image Classification**: Upload cricket images and get shot type predictions
- **Biomechanical Analysis**: Detailed analysis of batting technique
- **Pose Visualization**: Visual representation of detected pose landmarks
- **Real-time Processing**: Fast image processing and classification
- **Comprehensive Testing**: Full test suite for reliability

## API Endpoints

### Health Check
```
GET /
```
Returns API status and version information.

### Classify Cricket Shot
```
POST /api/classify
```
Upload an image and get cricket shot classification.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Cricket image to classify

**Response:**
```json
{
  "success": true,
  "prediction": {
    "shot_type": "drive",
    "confidence": 0.85,
    "probabilities": {
      "drive": 0.85,
      "legglance-flick": 0.08,
      "pullshot": 0.04,
      "sweep": 0.03
    }
  },
  "pose_analysis": {
    "landmarks_detected": 33,
    "features_extracted": 48
  },
  "processing_info": {
    "image_filename": "cricket_shot.jpg",
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

### Detailed Analysis
```
POST /api/analyze
```
Get detailed biomechanical analysis of cricket technique.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Cricket image to analyze

**Response:**
```json
{
  "success": true,
  "biomechanical_analysis": {
    "upper_body": {
      "left_shoulder_angle": 45.2,
      "right_elbow_angle": 90.1
    },
    "lower_body": {
      "left_hip_angle": 120.5,
      "right_knee_angle": 160.3
    }
  },
  "technique_recommendations": [
    {
      "category": "Stance",
      "recommendation": "Keep front foot close to the pitch of the ball",
      "importance": "high"
    }
  ]
}
```

### Model Information
```
GET /api/models
```
Get information about available models and shot classes.

## Installation

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
pip install -r api/requirements.txt
```

2. **Run the API:**
```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

## Usage Examples

### Using curl
```bash
# Classify a cricket shot
curl -X POST -F "image=@cricket_shot.jpg" http://localhost:5000/api/classify

# Get detailed analysis
curl -X POST -F "image=@cricket_shot.jpg" http://localhost:5000/api/analyze

# Check API health
curl http://localhost:5000/
```

### Using Python requests
```python
import requests

# Classify cricket shot
with open('cricket_shot.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/classify', files=files)
    result = response.json()
    print(f"Predicted shot: {result['prediction']['shot_type']}")
```

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)

## File Size Limits

- Maximum file size: 16MB
- Minimum image dimensions: 100x100 pixels
- Maximum image dimensions: 4000x4000 pixels

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid file or missing parameters
- `413 Payload Too Large`: File exceeds size limit
- `422 Unprocessable Entity`: Valid file but processing failed
- `500 Internal Server Error`: Server-side processing error

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run API-specific tests
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest tests/ --cov=api --cov-report=html
```

## Configuration

The API can be configured through `api/config.py`:

- Upload settings (file size limits, allowed extensions)
- Model configuration (paths, thresholds)
- Processing parameters (confidence thresholds, timeouts)

## Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
```

## Cricket Shot Classes

The model classifies images into four cricket shot types:

1. **Drive**: Straight bat shots played along the ground
2. **Legglance-flick**: Shots played to the leg side with wrist action
3. **Pullshot**: Horizontal bat shots to short-pitched balls
4. **Sweep**: Shots played with horizontal bat to spinning balls

## Architecture

The API uses the following components:

- **Flask**: Web framework for REST API
- **MediaPipe**: Pose estimation from images
- **PyTorch**: Deep learning model inference
- **OpenCV**: Image processing and validation
- **Custom Modules**: Cricket-specific feature extraction and analysis

## Logging

The API includes comprehensive logging for:
- Request/response tracking
- Error monitoring
- Performance metrics
- Processing statistics

Logs are written to `logs/api.log` with configurable levels.

