"""
Simple redirection from root app.py to the actual FastAPI application
"""
from runs.detect.yolov8m_kitti_enhanced.weights.main import app


# This file simply imports the actual app from the weights directory
# This allows GitHub Actions to find and deploy the correct app 