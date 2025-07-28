#!/usr/bin/env python3

"""
Server to listen for user events
"""

import sys
import os
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple


from shared_data import SharedData
from server_settings import *
from src.core.realtime import run_pipeline






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

    def check_shutdown(self):
        ...

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

        # Start the realtime pipeline
        self.server.start_realtime_pipeline()


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

            if self.path == WORKOUT_SESSION_STATE:
                for key in data:
                    self.server.shared_data.update(key, data[key])

            elif self.path == SET_MANUAL_EXERCISE:
                self.server.shared_data.update("current_selected_exercise", data["current_selected_exercise"])

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

    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass):

        """
        Constructor
        Args:
            server_address: a tuple which contains the IP address of the server (as a string) and the port (as an int)
            RequestHandlerClass: any class which extends the BaseHTTPRequestHandler class
        """

        super().__init__(server_address, RequestHandlerClass)

        # Shared data with the other threads
        self.shared_data = SharedData()

        # Default K-V pairs
        self.shared_data.update("paused", False)
        self.shared_data.update("stopped", False)

        # create realtime pipeline thread
        self.realtime_pipeline_thread = None
        self.reinit_realtime_pipeline_thread()

    def reinit_realtime_pipeline_thread(self):
        """
        Function to re-create the realtime pipeline thread for each new start
        """

        self.realtime_pipeline_thread = threading.Thread(target=run_pipeline, args=(self.shared_data,), daemon=False)

    def start_realtime_pipeline(self):
        """
        Function to start the realtime pipeline
        """

        # Check if the realtime pipeline is not already running
        #if self.shared_data.get("realtime_pipeline_finished"):
        # If this is not the first start of the pipeline, remove the realtime_pipeline_finished flag in shared_data
        self.shared_data.pop("realtime_pipeline_finished", None)
        self.shared_data.update("paused", False)
        self.shared_data.update("stopped", False)
        self.reinit_realtime_pipeline_thread()
        self.realtime_pipeline_thread.start()


def main():
    """
    Main function to start the server, which will start the realtime pipeline.
    """

    #try:
    server = Server((ADDRESS_SERVER, PORT_SERVER), HTTPRequestHandler)
    print(f"Server started on {ADDRESS_SERVER}:{PORT_SERVER}")
    server.serve_forever()

    #except Exception as ex:
     #   print(f"Error: {ex}", file=sys.stderr)



# Real case!
if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt: # Stop by pressing Ctrl+C
        pass
    finally:
        print("\n\nBye!")
        exit(0)