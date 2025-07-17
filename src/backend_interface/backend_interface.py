#!/usr/bin/env python3

"""
Frontend Interface Module

This module provides functions for interacting with the frontend interface.

"""

import sys
from typing import Dict
import json
import requests as rq
from .server import *


# By default in the frontend the port for backend_interface running is defined as 8080
# Check https://github.com/ixodev/magicmirror/blob/master/config/config.js
PORT = 8080
# Has also to be synced with the frontend
# Check https://github.com/ixodev/magicmirror/blob/master/modules/MMM-WorkoutTracker/node_helper.js
ROUTE = "workout-tracking"
OK = 200


def update_workout_state(total_reps: Dict[str, int], current_exercise: str) -> None:
    """
    Update the frontend with current workout state.
    
    This is a placeholder function that will be implemented by the frontend team.
    It should update the frontend interface with the current workout progress.
    
    Args:
        total_reps: Dictionary mapping exercise names to total rep counts
                   Example: {'push-ups': 5, 'squats': 3, 'pull-ups': 0, 'dips': 2}
        current_exercise: Currently detected exercise (or 'unknown' if none detected)
                         Example: 'push-ups', 'squats', 'pull-ups', 'dips', 'unknown'
    
    Returns:
        None

    """

    # Create JSON structure for frontend
    workout_state = {
        "current_exercise": current_exercise,
        # If current_exercise is 'unknown' just send 0 total reps to the mirror
        "total_reps": total_reps.get(current_exercise) if current_exercise != "unknown" else '0',
        "timestamp": None  # Can be added if needed
    }

    # Make a POST request to the frontend with workout_state as JSON payload
    rep = rq.post(f"http://localhost:{PORT}/{ROUTE}", json=workout_state)

    if rep.status_code != OK:
        print(f"Error: could not POST to localhost:{PORT}", file=sys.stderr)



