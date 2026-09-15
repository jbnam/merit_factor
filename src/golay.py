"""
Golay Merit Factors Main Module
===============================

An argument parser for the Cross-Entropy Method (CEM)
applied to Golay's Merit Factor Problem, with comprehensive validation,
logging, and configuration management.

This module provides:
- Argument parsing with type validation
- Configuration validation and error handling
- Type-safe configuration data classes
- Comprehensive documentation and examples

Author: MJMARI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Argument Validation:
  - All arguments are validated after parsing
  - Type conversions are checked for correctness
  - Range constraints are enforced
  - Mutual dependencies are validated

Error Handling:
  - Clear error messages for invalid arguments
  - Suggests valid ranges for out-of-range values
  - Logs all validation failures
"""
import os

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

import torch

import common as cm

if __name__ == "__main__":
    # Set GPU environment
    if torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = cm.STR_CC

    #
