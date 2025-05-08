#!/bin/bash
cd /home/site/wwwroot
cp -f runs/detect/yolov8m_kitti_enhanced/weights/main.py app.py
mkdir -p uploads static templates
pip install -r runs/detect/yolov8m_kitti_enhanced/weights/requirements.txt
gunicorn --bind=0.0.0.0 --timeout 600 app:app 