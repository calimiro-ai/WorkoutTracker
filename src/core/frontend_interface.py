#!/usr/bin/env python3
"""
Frontend Interface Module

This module provides functions for interacting with the frontend interface.
Currently contains placeholder functions that will be implemented by the frontend team.
"""

import sys
from typing import Dict
import json
import requests as rq


# By default in the frontend the port for server running is defined as 8080
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
        two booleans: the first one indicates if the user paused the workout session,
        the second indicates if the user stopped the workout session
    """
    print(f"Attempt to send data to the mirror, update_workout_state()...")

    # Create JSON structure for frontend
    workout_state = {
        "current_exercise": current_exercise,
        # If current_exercise is 'unknown' just send 0 total reps to the mirror
        "total_reps": total_reps.get(current_exercise) if current_exercise != "unknown" else '0',
        "timestamp": None  # Can be added if needed
    }

    print(f"Making POST request to \"http://localhost:{PORT}/{ROUTE}\"...")
    # Make a POST request to the frontend with workout_state as JSON payload
    rep = rq.post(f"http://localhost:{PORT}/{ROUTE}", json=workout_state)

    if rep.status_code != OK:
        print(f"Error: could not POST to localhost:{PORT}", file=sys.stderr)

    try:
        print(f"Response:\n{rep.text}")
        print("Receiving & serializing response...")
        result = json.loads(rep.text)
    except Exception as ex:
        print(f"Error: could not deserialize JSON, reason: {ex}", file=sys.stderr)
        # Default values for paused & stopped are both False
        return False, False

    paused = result.get("paused") # is the workout session paused?
    stopped = result.get("stopped") # is the workout session stopped?

    if paused is None or stopped is None:
        return False, False

    return paused, stopped


