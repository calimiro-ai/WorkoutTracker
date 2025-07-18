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
ROUTE = "/workout-tracking"
GET_AVAILABLE_EXERCISES = "/available-exercises"
MANUAL_EXERCISE = "/set-manual-exercise"

# Store the different routes
GET_REQUESTS = [GET_AVAILABLE_EXERCISES]
POST_REQUESTS = [ROUTE, MANUAL_EXERCISE]

# Status codes
OK = 200
INVALID = 400
NOT_FOUND = 404