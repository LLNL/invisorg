"""
Basic utilities for logging, IO, etc.
"""

import sys
import time
import os
import warnings
import csv
import glob
from datetime import timedelta
from enum import Enum

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------- Functions & Methods --------------------------------------------------
#
# region Functions & Methods

def get_print_time():
    """
    Helper function for returning a formatted date-time string for stdout messages.

    :return: The current time in the format: %d-%b-%Y %H:%M:%S
    """
    return time.strftime('%d-%b-%Y %H:%M:%S', time.localtime())


def print_develop_message():
    """
    Print a development mode warning message to STDOUT with the word 'WARNING' in red.
    The message includes the current timestamp (using get_print_time()), a warning emoji, and
    the word 'WARNING' styled in red using ANSI escape codes.

    Note: \033[91m starts red text; \033[0m resets the color.
    """
    print(f"[{get_print_time()}] ⚠️  \033[91mWARNING: Running in Develop Mode!\033[0m ⚠️")


def create_dir(with_path='default_path'):
    """
    Create a directory if it does not exist.

    Args:
        with_path (str): The directory path to create.

    Returns:
        str: The directory path.
    """
    if not os.path.exists(with_path):
        os.makedirs(with_path)
    return with_path


def restricted_float(x):
    """
    Used by the arguments to check the learning rate.

    :param x: The input argument to check for float type.
    :raises argparse.ArgumentTypeError: Raised if the input is not a floating point.
    :raises argparse.ArgumentTypeError: Raised if the input is outside of a range.
    :return: The input argument if it passed all the checks.
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def test_size_float(x):
    """
    Used by the CLI to check the holdout test set size fraction.

    This function ensures that the input can be converted to a float and is within the
    range [0.0, 1.0]. The test size represents the fraction of the dataset reserved for testing.

    :param x: The input argument to check for float type.
    :raises argparse.ArgumentTypeError: If the input is not a floating-point literal.
    :raises argparse.ArgumentTypeError: If the input is outside the range [0.0, 1.0].
    :return: The input argument converted to a float if it passes all checks.
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x
