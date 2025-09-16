#!/bin/bash

# Navigate to the app directory to ensure git commands work correctly
cd "$(dirname "$0")" || exit

# Pull the latest changes from GitHub
echo "Pulling latest changes from GitHub..."
git pull origin main

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Stop any existing app instance to prevent conflicts
if [ -f app.pid ]; then
    echo "Stopping existing app instance..."
    kill "$(cat app.pid)" 2>/dev/null
    rm app.pid
fi

# Run the Streamlit app in the background
echo "Starting Streamlit app..."
nohup python3 -m streamlit run app.py --server.port=$STREAMLIT_SERVER_PORT &> app.log &

# Get the process ID of the new instance
echo $! > app.pid

echo "App is running in the background on port $STREAMLIT_SERVER_PORT"
echo "Check app.log for output"
echo "To stop the app, run: kill $(cat app.pid) && rm app.pid"
