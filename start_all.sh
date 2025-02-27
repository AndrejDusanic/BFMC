#!/bin/bash

# Kill any lingering processes that include "main.py" in their command line.
echo "Killing lingering main.py processes..."
pkill -f main.py

# Start main.py in the background and capture its PID
python3 main.py &
MAIN_PID=$!

# Wait for 20 seconds
sleep 20

# Start pid.py in the background and capture its PID
sudo python3 ~/newBrain/BFMC/src/hardware/camera/threads/pid.py &
PID_PID=$!

# Define a cleanup function that kills the processes in the desired order
cleanup() {
    echo "Terminating main.py (PID: $MAIN_PID)..."
    kill $MAIN_PID 2>/dev/null
    sleep 10  # Optional: give it a moment to shut down gracefully
    echo "Terminating pid.py (PID: $PID_PID)..."
    kill $PID_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to trigger the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for both processes to finish
wait
