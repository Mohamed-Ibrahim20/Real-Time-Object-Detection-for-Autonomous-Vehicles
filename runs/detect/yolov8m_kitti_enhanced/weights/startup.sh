#!/bin/bash
cd /home/site/wwwroot

# Copy main application and model
cp -f runs/detect/yolov8m_kitti_enhanced/weights/main.py app.py
cp -f runs/detect/yolov8m_kitti_enhanced/weights/best.pt best.pt

# Create expected directories
mkdir -p uploads
mkdir -p static
mkdir -p templates

# Copy static and template files from their deployed location to the root static/templates
# Check if source directories exist before copying to avoid errors if they are not present
if [ -d "runs/detect/yolov8m_kitti_enhanced/weights/static" ]; then
  cp -rf runs/detect/yolov8m_kitti_enhanced/weights/static/* static/
fi

if [ -d "runs/detect/yolov8m_kitti_enhanced/weights/templates" ]; then
  cp -rf runs/detect/yolov8m_kitti_enhanced/weights/templates/* templates/
fi

# Install dependencies
pip install -r runs/detect/yolov8m_kitti_enhanced/weights/requirements.txt

# Start Gunicorn
gunicorn --bind=0.0.0.0 --timeout 600 app:app --log-level debug 