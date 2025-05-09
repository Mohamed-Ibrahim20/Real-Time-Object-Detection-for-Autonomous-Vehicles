#!/bin/bash

# Check Python version
echo "Using Python version:"
python3 --version

# Clean up any existing virtual environment
rm -rf /home/site/wwwroot/antenv

# Set up virtual environment
python3 -m venv /home/site/wwwroot/antenv

# Activate the virtual environment
source /home/site/wwwroot/antenv/bin/activate

# Install dependencies from requirements.txt
pip install -r /home/site/wwwroot/requirements.txt

# Start the application using Gunicorn with Uvicorn worker
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0 --timeout 600
