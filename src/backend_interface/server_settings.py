#!/usr/bin/env python3

"""
Server definitions
"""


# Address
ADDRESS_SERVER = "0.0.0.0"

# Port of the server
PORT_SERVER = 8000

# By default in the frontend the port for backend_interface running is defined as 8080
# Check https://github.com/ixodev/magicmirror/blob/master/config/config.js
PORT_FRONTEND = 8080

# Has to be synced with the frontend
# Check https://github.com/ixodev/magicmirror/blob/master/modules/MMM-WorkoutTracker/node_helper.js
WORKOUT_SESSION_STATE = "/workout-session-state"
WORKOUT_SESSION_START = "/workout-session-start"
WORKOUT_LOADING_EXERCISES_FINISHED = "/workout-loading-exercises-finished"
SET_MANUAL_EXERCISE = "/set-manual-exercise"

# Store the different routes
GET_REQUESTS = [WORKOUT_SESSION_START]
POST_REQUESTS = [WORKOUT_SESSION_STATE, SET_MANUAL_EXERCISE]

# Status codes
OK = 200
INVALID = 400
NOT_FOUND = 404