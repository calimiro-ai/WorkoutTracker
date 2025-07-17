#!/usr/bin/env python3

"""
Thread-safe dictionary-like data structure to help threads to communicate with each other
"""

from collections.abc import Hashable
from typing import Any
import threading


class SharedData:
    """
    Thread-safe dictionary-like data structure to help threads to communicate with each other
    """

    def __init__(self):
        # The data structure which will hold the shared data between the threads
        self.data = {}
        # The mutex to impeach race conditions
        self.lock = threading.Lock()

    def update(self, k: Hashable, v: Any):
        """
        Updates the dict with a new K-V pair, thread-safe
        If the specified key exists already in the dict, the dedicated K-V pair will simply be updated

        Args:
            k: the key must be an instance of a class which implements the Hashable abstract class (a hashable object)
            v: the value can be any object type
        Returns:
            None
        """

        with self.lock:
            self.data.update({k: v})

    def get(self, k: Hashable):
        """
        Gets a value from the dict by its dedicated key

        Args:
            k: the key must be an instance of a class which implements the Hashable abstract class (a hashable object)
        Returns:
            None if the specified key was not part of a K-V pair in the dict
            The dedicated value if the specified key is associated with a value as a K-V pair in the dict
        """

        # No need for lock because reading on a dict is not an atomic operation in CPython
        return self.data.get(k)

    def __str__(self):
        """
        String representation of the data structure

        Args:
            No args
        Returns:
            str
        """

        string_builder = "{\n"

        for k in self.data:
            string_builder += f"{k}: {self.data.get(k)}\n"

        string_builder += "}"

        return string_builder