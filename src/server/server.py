"""
Simple server to communicate with the frontend
"""

import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


# Address
ADDRESS_SERVER = "0.0.0.0"

# Port of the server
PORT_SERVER = 8000

# By default in the frontend the port for server running is defined as 8080
# Check https://github.com/ixodev/magicmirror/blob/master/config/config.js
PORT_FRONTEND = 8080

# Has to be synced with the frontend
# Check https://github.com/ixodev/magicmirror/blob/master/modules/MMM-WorkoutTracker/node_helper.js
ROUTE = "/workout-tracking"


# Status codes
OK = 200
INVALID = 400
NOT_FOUND = 404


class HTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Server to handle POST requests from the frontend.
    """

    def do_POST(self):
        """
        Override method to handle POST requests from the frontend interface
        """

        if self.path != ROUTE:
            response = "Error: not found".encode("utf-8")
            self.send_response(NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return


        # Determine the amount of bytes to read
        content_length = int(self.headers["Content-Length"])

        # Get the body and decode it
        post_body = self.rfile.read(content_length).decode("utf-8")

        try:

            # Try to deserialize json payload
            data = json.loads(post_body)

        except json.JSONDecodeError as ex:

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

        # Doing some stuff with the body of the request
        print(data)



# Test case
if __name__ == "__main__":
    server = HTTPServer((ADDRESS_SERVER, PORT_SERVER), HTTPRequestHandler)
    print(f"Server started on {ADDRESS_SERVER}:{PORT_SERVER}")
    server.serve_forever()

