# Cricket Shot Classification from Images

This extension to the Cricket Biomechanics Analysis system adds support for classifying cricket shots from static images using pose estimation and machine learning.

## Overview

The image-based cricket shot classification system works by:

1. Extracting pose landmarks from cricket shot images using MediaPipe
2. Calculating joint angles and other biomechanical features from the pose data
3. Training a multiclass LSTM model to classify different cricket shots
4. Providing prediction capabilities for new images

## Directory Structure

```
Biomechanics/
├── configs/
│   └── config.yaml           # Configuration file
├── data/
│   ├── raw/
│   │   └── images/           # Raw image dataset organized by shot type
│   │       ├── cover_drive/
│   │       ├── straight_drive/
│   │       ├── pull_shot/
│   │       └── ...
│   └── processed/
│       ├── poses/            # Extracted pose data (JSON)
│       └── features/         # Calculated features (CSV)
│           └── splits/       # Train/val/test splits
├── models/                   # Trained models
├── results/                  # Prediction results
└── src/
    ├── data_collection/
    │   └── image_pose_extractor.py  # Image pose extraction
    ├── feature_engineering/
    │   ├── joint_angles.py          # Feature calculation
    │   └── dataset_preparer.py      # Dataset preparation
    ├── model/
    │   └── multiclass_lstm_model.py # LSTM model for classification
    ├── train_image_classifier.py    # Training script
    └── predict_cricket_shot.py      # Prediction script
```

## Setup

1. Organize your cricket shot image dataset in the `data/raw/images` directory, with subdirectories for each shot type:

```
data/raw/images/
├── cover_drive/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── straight_drive/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Poses and Train the Model

To extract poses from images and train the classifier:

```bash
python src/train_image_classifier.py --extract_poses --visualize
```

Options:
- `--config`: Path to configuration file (default: `configs/config.yaml`)
- `--data_dir`: Directory containing the image dataset (default: `data/raw/images`)
- `--output_dir`: Directory to save the trained model (default: `models`)
- `--pose_dir`: Directory to save/load extracted pose data (default: `data/processed/poses`)
- `--features_dir`: Directory to save/load extracted features (default: `data/processed/features`)
- `--extract_poses`: Extract poses from images before training
- `--visualize`: Save pose visualizations
- `--model_name`: Name of the model file to save (default: `cricket_shot_classifier.pt`)
- `--subset`: Use a subset of images per class for initial testing (e.g., `--subset 10`)

### 2. Predict Cricket Shots from New Images

To predict cricket shots from new images using a trained model:

```bash
python src/predict_cricket_shot.py --model models/cricket_shot_classifier.pt --input path/to/image.jpg --visualize
```

Options:
- `--config`: Path to configuration file (default: `configs/config.yaml`)
- `--model`: Path to the trained model file (required)
- `--input`: Path to input image or directory of images (required)
- `--output_dir`: Directory to save results (default: `results`)
- `--visualize`: Visualize pose and prediction on the image

## Example Workflow

1. **Initial Testing with a Subset**

   Start by training on a small subset to verify the approach:

   ```bash
   python src/train_image_classifier.py --extract_poses --visualize --subset 10
   ```

2. **Full Training**

   Once the approach is validated, train on the full dataset:

   ```bash
   python src/train_image_classifier.py --extract_poses --visualize
   ```

3. **Prediction**

   Use the trained model to classify new images:

   ```bash
   python src/predict_cricket_shot.py --model models/cricket_shot_classifier.pt --input data/test_images --visualize
   ```

## Model Performance

After training, you can find evaluation metrics and visualizations in the model output directory:

- `confusion_matrix.png`: Confusion matrix showing classification performance
- `training_history.png`: Training and validation loss/accuracy curves
- `evaluation_metrics.json`: Detailed classification metrics

## Extending the System

To add support for new cricket shot types:

1. Add a new subdirectory with images in `data/raw/images/`
2. Retrain the model using the training script

The system will automatically detect and include the new shot type in the classification model.

## Integration with Video Analysis

This image-based classification system complements the existing video analysis pipeline. While the video analysis focuses on temporal dynamics and technique assessment, the image classifier provides quick shot identification from static images.

## Sprint 2 Updates

### New Features

- **Multiclass Classification**: Extended the system to support multiple cricket shot types beyond just straight drives
- **Improved Feature Engineering**: Enhanced joint angle calculations for better shot differentiation
- **Batch Processing**: Added support for processing directories of images for both training and prediction
- **Visualization Enhancements**: Improved pose landmark visualization and prediction confidence display
- **Performance Optimizations**: Reduced processing time for pose extraction and feature calculation

### Technical Improvements

- Refactored code for better modularity and reusability
- Added comprehensive error handling for missing landmarks and invalid poses
- Improved dataset preparation with stratified train/validation/test splits
- Enhanced model evaluation with detailed metrics and visualizations
- Added support for saving and loading model metadata for easier deployment