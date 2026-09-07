"""
Data Types and Global Constants
===============================

Collection of the user defined data types and global constants for
the Cross-Entropy Method (CEM) applied to Golay's Merit Factor Problem

This module provides:
- Definitions of project specific data types for NumPy, PyTorch, Python, and CUDA
- Global constants for the CEM algorithm

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
The maximum number of 32-bit registers per thread on 1080 Ti is 255.
Hence, the maximum number of registers available for a binary sequence per thread is
255 - 32 = 223. (There are extra 32 regeisters for local variables in the CUDA kernel)
"""

import torch
import numpy as np

# ============================================================================
# Global Constants
# ============================================================================

# The number of bits in a limb (32 bits for 32-bit integers)
LIMB_BIT_SIZE = 32

# The minimum length of binary sequences of 0s and 1s for merit factor (16 bits)
MIN_BIT_LEN = 16

# The maximum length of binary sequences of 0s and 1s for merit factor (4096 bits)
MAX_BIT_LEN = 4096 # 2^{12} < 223 * LIMB_BIT_SIZE

# The maximum number of limbs for a binary sequence of 0s and 1s for merit factor
MAX_NUM_LIMBS = 128 # MAX_BIT_LEN / LIMB_BIT_SIZE

# The maximum number of threads per block in CUDA kernel
# This can be vary depending on the GPU architecture,
# but 256 is a safe choice for most modern GPUs
MAX_BLOCK_SIZE = 256

# The number of registers per thread required for computing the merit factor of
# a binary sequence of 0s and 1s in the CUDA kernel in merit_factor_gpu.py.
NUM_REG_PER_THREAD = 32

# The number of threads per warp in CUDA.
WARP_SIZE = 32

# A mask for the maximum value of a 32-bit unsigned integer, for bitwise operations
MAX_UINT32 = 0xFFFFFFFF
MSB_ZERO_MASK32 = 0x7FFFFFFF  # Binary: 0111...1111
MSB_ONE_MASK32 = 0x80000000   # Binary: 1000...0000

MAX_NUM_EPOCHS = 10000
MAX_SAMPLE_SIZE = 10000000

# ============================================================================
# Data Types
# ============================================================================

T_UINT32 = torch.uint32
T_INT32 = torch.int32
T_INT64 = torch.int64
T_FLOAT32 = torch.float32
T_FLOAT64 = torch.float64

N_UINT32 = np.uint32
N_INT32 = np.int32
N_INT64 = np.int64
N_FLOAT32 = np.float32
N_FLOAT64 = np.float64

# ============================================================================
# File Paths
# ============================================================================

LOG_FILE_NAME = "experiments.log"
SUMMARY_BIN_SEQ_FILE_NAME = "summary_bin_seq.hdf5"
SUMMARY_ELITES_FILE_NAME = "summary_elites.hdf5"
DATA_ELITES_FILE_NAME = "data_elites.hdf5"

# ============================================================================
# Conguration Options
# ============================================================================
