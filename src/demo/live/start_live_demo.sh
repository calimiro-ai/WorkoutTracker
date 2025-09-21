#!/bin/bash
# Live Workout Tracker Demo Startup Script

echo "=== Live Workout Tracker Demo ==="
echo "Starting live camera demo..."
echo

# Check if model exists
MODEL_PATH="models/final/best_model.keras"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure you have a trained model in the models/final/ directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "Model found: $MODEL_PATH"
echo "Starting live demo..."
echo
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 'r' to reset rep counter"
echo "  - Make sure you have a camera connected"
echo

# Activate virtual environment and run demo
source venv/bin/activate
python src/demo/live/live_demo.py --model "$MODEL_PATH" --camera 0 --inference-rate 3 --peak-delay 2
