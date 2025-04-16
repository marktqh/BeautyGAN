# Data Directory

This directory should contain the paired face images used for training and testing the beautification model.

## Directory Structure

```
data/
├── raw/      # Original (before) images
└── aft/      # Beautified (after) images
```

## Data Requirements

- Both directories should contain corresponding images with the same filenames or numerical indices (e.g., `001.jpg` in both directories)
- Images should be high-resolution facial photographs
- Supported image formats: .jpg, .JPG, .jpeg, .png, .webp, .arw, .cr2

## Note

The actual image data is not included in this repository due to privacy and copyright considerations. You'll need to provide your own paired before/after face images for training the model.

## Sample Images

For testing purposes, you should place at least the following images in both directories:
- 1201.JPG - Used for training visualization
- 3401.jpg - Used for testing visualization
