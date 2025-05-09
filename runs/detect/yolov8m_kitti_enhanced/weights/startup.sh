#!/bin/bash
echo "Starting startup.sh script..."

cd /home/site/wwwroot
echo "Current directory: $(pwd)"

echo "Listing files in wwwroot before operations:"
ls -R

# Copy main application and model
echo "Copying main.py to app.py..."
cp -f runs/detect/yolov8m_kitti_enhanced/weights/main.py app.py
echo "Copying best.pt to root..."
cp -f runs/detect/yolov8m_kitti_enhanced/weights/best.pt best.pt

# Create expected directories
echo "Creating directories (uploads, static, templates)..."
mkdir -p uploads
mkdir -p static
mkdir -p templates

# Copy static and template files
echo "Copying static files..."
if [ -d "runs/detect/yolov8m_kitti_enhanced/weights/static" ]; then
  cp -rfv runs/detect/yolov8m_kitti_enhanced/weights/static/* static/
  echo "Static files copied."
else
  echo "WARNING: Source static directory not found!"
fi

echo "Copying template files..."
if [ -d "runs/detect/yolov8m_kitti_enhanced/weights/templates" ]; then
  cp -rfv runs/detect/yolov8m_kitti_enhanced/weights/templates/* templates/
  echo "Template files copied."
else
  echo "WARNING: Source templates directory not found!"
fi

echo "Listing files in wwwroot after copy operations:"
ls -R

# Install dependencies
echo "Installing dependencies from runs/detect/yolov8m_kitti_enhanced/weights/requirements.txt..."
pip install -r runs/detect/yolov8m_kitti_enhanced/weights/requirements.txt
echo "Dependency installation finished."

# Start Gunicorn
echo "Starting Gunicorn..."
# Redirect Gunicorn logs to a file for easier debugging via Kudu/FTP
gunicorn --bind=0.0.0.0 --timeout 600 app:app --log-level debug --access-logfile /home/LogFiles/gunicorn_access.log --error-logfile /home/LogFiles/gunicorn_error.log

echo "Gunicorn command executed." 