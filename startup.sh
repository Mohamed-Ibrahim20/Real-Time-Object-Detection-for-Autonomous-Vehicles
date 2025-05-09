#!/bin/bash
echo "Using Python version:"
python3 --version

# Set up virtual environment
python3 -m venv /home/site/wwwroot/antenv

# Activate the virtual environment
source /home/site/wwwroot/antenv/bin/activate

# Install dependencies from requirements.txt
pip install -r /home/site/wwwroot/requirements.txt

# Start the application using Gunicorn or your desired server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0 --timeout 600
