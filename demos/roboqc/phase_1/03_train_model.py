#!/usr/bin/env python3
"""Model training entry point (RoboQC)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from roboqc.intent.trainer import main as train_main

if __name__ == "__main__":
    train_main()
