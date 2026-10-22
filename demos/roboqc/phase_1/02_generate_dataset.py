#!/usr/bin/env python3
"""Dataset generator entry point (RoboQC)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from roboqc.dataset.generator import main as gen_main

if __name__ == "__main__":
    gen_main()
