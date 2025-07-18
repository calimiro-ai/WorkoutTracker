#!/usr/bin/env python3

"""
Backend interface to communicate with the frontend
"""

import sys
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple

from backend_interface.shared_data import SharedData



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


class HTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Server to handle POST requests from the frontend.
    """

    def send_404(self):
        """
        Method to send 404 as status code and an error message as a response for an invalid request
        """
        response = "Error: not found".encode("utf-8")
        self.send_response(NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


    def do_GET(self):
        """
        Override method to handle GET requests from the frontend interface
        """
        if not self.path in GET_REQUESTS:
            self.send_404()
            return

        available_models = []

        # Search in the models directory
        for file in os.listdir("models/segmentation"):
            if file.endswith(".keras"):
                available_models.append(file.split(".")[0]) # Remove extension of the file

        response = json.dumps({"available_exercises": available_models})

        self.send_response(OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))


    def do_POST(self):
        """
        Override method to handle POST requests from the frontend interface
        """

        if not self.path in POST_REQUESTS:
            self.send_404()
            return


        # Determine the amount of bytes to read
        content_length = int(self.headers["Content-Length"])

        # Get the body and decode it
        post_body = self.rfile.read(content_length).decode("utf-8")

        try:
            # Try to deserialize json payload
            data = json.loads(post_body)

            for key in data:
                self.server.shared_data.update(key, data[key])

        except Exception as ex:

            # Error (JSON could not be deserialized)
            response = "Invalid JSON".encode("utf-8")
            self.send_response(INVALID)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

            # Log to the console + send to the client who made the request
            print(f"Error: {ex}", file=sys.stderr)
            self.wfile.write(f"Error: {ex}".encode("utf-8")) # Encode the response

            # Nothing more to do
            return


        # Send response
        response = "OK".encode("utf-8")
        self.send_response(OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)



class Server(HTTPServer):

    """
    HTTP Server
    Custom class for an HTTP Server to pass a thread-safe SharedData object to the HTTPRequestHandler
    """

    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass, shared_data: SharedData):

        """
        Constructor
        Args:
            server_address: a tuple which contains the IP address of the server (as a string) and the port (as an int)
            RequestHandlerClass: any class which extends the BaseHTTPRequestHandler class
            shared_data: a SharedData object
        """

        super().__init__(server_address, RequestHandlerClass)
        self.shared_data = shared_data



def backend_process(shared_data: SharedData):
    """
    Target function for the server thread.
    Args:
        shared_data: a dictionary which contains all the data that will be shared between this thread and the main thread.
    Returns:
        None
    """

    try:
        server = Server((ADDRESS_SERVER, PORT_SERVER), HTTPRequestHandler, shared_data)
        print(f"Server started on {ADDRESS_SERVER}:{PORT_SERVER}")
        server.serve_forever()
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)



# Test case
if __name__ == "__main__":
    try:
        # No shared_data required, this is just a test case
        backend_process(SharedData())
    except KeyboardInterrupt: # Stop by pressing Ctrl+C
        print("Bye!")
        exit(0)