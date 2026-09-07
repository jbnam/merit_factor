"""
Cross-Entropy Method (CEM) Initialization Module
=================================================

A high-performance module for initializing the CEM algorithm for Golay's Merit
Factor Problem with comprehensive logging, configuration validation, and data
management.

This module provides:
- Robust CEM initialization with hyperparameter validation
- Professional logging with rotating file handlers
- Distribution initialization (naive or recursive methods)
- Data persistence with HDF5 format
- Performance optimization and memory management

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Distribution Methods:
  - Naive: Initialize with uniform 0.5 probability
  - Recursive: Load from previous run with smaller sequence length

File Structure:
  - logs/{method_dist}/{len_bin_seq}/{timestamp}/
    - logs.log: Main log file
  - logs/{method_dist}/{len_bin_seq}/
    - summary_bin_seq.hdf5: All sequences statistics
    - summary_elites.hdf5: Elite sequences statistics
    - data_elites.hdf5: Elite sequence data
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import h5py

from src.common import cm

# ============================================================================
# Configuration Data Classes
# ============================================================================

class DistributionMethod(Enum):
    """Enumeration for distribution initialization methods.

    Attributes
    ----------
    NAIVE : str
        Initialize with uniform 0.5 probability.
    RECURSIVE : str
        Load from previous run with smaller sequence length.
    """
    NAIVE = "naive"
    RECURSIVE = "recursive"

@dataclass
class CEMInitConfig:
    """Configuration for CEM initialization.

    Attributes
    ----------
    len_bin_seq : int
        Length of binary sequences (16-4096).
    method_dist : str
        Distribution method ("naive" or "recursive").
    num_limbs : int
        Number of limbs per sequence.
    """
    len_bin_seq: int
    method_dist: str
    num_limbs: int

    def __post_init__(self):
        """Validate configuration."""
        self._validate_len_bin_seq()
        self._validate_method_dist()
        self._validate_num_limbs()

    def _validate_len_bin_seq(self) -> None:
        """Validate sequence length."""
        # WARNING! len_bin_seq must be in [16, 4096] due to memory/algorithm constraints
        if not isinstance(self.len_bin_seq, int):
            raise TypeError(f"len_bin_seq must be int, got {type(self.len_bin_seq)}")

        if self.len_bin_seq < 16 or self.len_bin_seq > 4096:
            raise ValueError(
                f"len_bin_seq must be in [16, 4096], got {self.len_bin_seq}"
            )

    def _validate_method_dist(self) -> None:
        """Validate distribution method."""
        # WARNING! method_dist must be one of predefined methods
        valid_methods = [m.value for m in DistributionMethod]

        if self.method_dist not in valid_methods:
            raise ValueError(
                f"method_dist must be one of {valid_methods}, "
                f"got '{self.method_dist}'"
            )

    def _validate_num_limbs(self) -> None:
        """Validate number of limbs."""
        # WARNING! num_limbs must be positive and match len_bin_seq
        if not isinstance(self.num_limbs, int):
            raise TypeError(f"num_limbs must be int, got {type(self.num_limbs)}")

        if self.num_limbs <= 0:
            raise ValueError(f"num_limbs must be positive, got {self.num_limbs}")

        # Verify consistency with len_bin_seq
        expected_limbs = (self.len_bin_seq + 31) // 32
        if self.num_limbs != expected_limbs:
            raise ValueError(
                f"num_limbs {self.num_limbs} doesn't match expected {expected_limbs} "
                f"for len_bin_seq={self.len_bin_seq}"
            )

@dataclass
class CEMInitResult:
    """Result of CEM initialization.

    Attributes
    ----------
    distribution : torch.Tensor
        Initial Bernoulli distribution.
    log_dir : str
        Logging directory path.
    summary_bin_seq_path : str
        Path to binary sequence summary file.
    summary_elites_path : str
        Path to elites summary file.
    data_elites_path : str
        Path to elites data file.
    max_merit_dict : dict
        Dictionary of elite sequences with maximum merit factor.
    logger : logging.Logger
        Configured logger instance.
    """
    distribution: torch.Tensor
    log_dir: str
    summary_bin_seq_path: str
    summary_elites_path: str
    data_elites_path: str
    max_merit_dict: Dict[str, Any]
    logger: logging.Logger


# ============================================================================
# CEM Initializer
# ============================================================================

class CEMInitializer:
    """Professional CEM initializer with robust configuration management.

    Parameters
    ----------
    base_dir : str, optional
        Base directory for logs. Default is current directory.

    Examples
    --------
    >>> initializer = CEMInitializer()
    >>> result = initializer.initialize(len_bin_seq=256, method_dist="naive", num_limbs=8)
    """

    def __init__(self, base_dir: str = "."):
        """Initialize CEM initializer."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _create_log_directory(
        self,
        method_dist: str,
        len_bin_seq: int
    ) -> Tuple[str, str]:
        """
        Create log directory structure and return paths.

        Parameters
        ----------
        method_dist : str
            Distribution method.
        len_bin_seq : int
            Sequence length.

        Returns
        -------
        tuple of (str, str)
            (timestamp_log_dir, method_len_dir)

        WARNING! Directory structure must be: logs/{method}/{len_seq}/{timestamp}/
        WARNING! Timestamp format must be ISO format with proper escaping
        """
        # Create method/length directory structure
        method_dir = self.base_dir / "logs" / method_dist
        method_dir.mkdir(parents=True, exist_ok=True)

        method_len_dir = method_dir / str(len_bin_seq)
        method_len_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        # WARNING! Use %H:%M:%S not $H-$M-%S, and %Y-%m-%d not strfttime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_log_dir = method_len_dir / timestamp
        timestamp_log_dir.mkdir(parents=True, exist_ok=True)

        return str(timestamp_log_dir), str(method_len_dir)

    def _initialize_naive_distribution(
        self,
        len_bin_seq: int,
        logger: logging.Logger
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Initialize naive distribution (uniform 0.5 probability).

        Parameters
        ----------
        len_bin_seq : int
            Sequence length.
        logger : logging.Logger
            Logger instance.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distribution, max_merit_dict)

        Examples
        --------
        >>> dist, max_dict = initializer._initialize_naive_distribution(256, logger)
        """
        logger.info("Initializing naive distribution (uniform 0.5)")

        # WARNING! Create distribution with correct dtype and value
        distribution = torch.full(
            (len_bin_seq,),
            0.5,
            dtype=cm.T_FLOAT32
        )

        # Initialize empty max merit dictionary
        max_merit_dict = {
            "bin_seq_2d": torch.empty((1, (len_bin_seq + 31) // 32), dtype=cm.T_INT32),
            "merit_factor": 0.0
        }

        logger.info(f"Initialized naive distribution: shape={distribution.shape}")

        return distribution, max_merit_dict

    def _initialize_recursive_distribution(
        self,
        method_dist_dir: str,
        len_bin_seq: int,
        logger: logging.Logger
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Initialize recursive distribution from previous run.

        Parameters
        ----------
        method_dist_dir : str
            Method distribution directory.
        len_bin_seq : int
            Target sequence length.
        logger : logging.Logger
            Logger instance.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distribution, max_merit_dict)

        Raises
        ------
        ValueError
            If no previous run found.

        WARNING! Must find directories with numeric names (sequence lengths)
        WARNING! Must load most recent checkpoint from found directory
        """
        logger.info(f"Initializing recursive distribution from {method_dist_dir}")

        # WARNING! Scan for directories with numeric names (sequence lengths)
        method_dir = Path(method_dist_dir)
        if not method_dir.exists():
            raise ValueError(f"Method directory not found: {method_dist_dir}")

        # Find all numeric subdirectories (previous sequence lengths)
        numeric_dirs = []
        for item in method_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                numeric_dirs.append((int(item.name), item))

        if not numeric_dirs:
            raise ValueError(
                f"No previous runs found in {method_dist_dir}. "
                f"Use 'naive' method instead."
            )

        # Find the maximum length that is less than len_bin_seq
        # WARNING! Must find largest len_bin_seq that is < current len_bin_seq
        valid_dirs = [(length, path) for length, path in numeric_dirs if length < len_bin_seq]

        if not valid_dirs:
            raise ValueError(
                f"No previous run with sequence length < {len_bin_seq}. "
                f"Use 'naive' method instead."
            )

        max_length, source_dir = max(valid_dirs, key=lambda x: x[0])

        logger.info(f"Found previous run with len_bin_seq={max_length}")

        # Try to load from most recent timestamp directory
        # WARNING! Must search for most recent timestamp directory
        timestamp_dirs = sorted(
            [d for d in source_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )

        if not timestamp_dirs:
            raise ValueError(f"No timestamp directories found in {source_dir}")

        latest_timestamp_dir = timestamp_dirs[0]

        # Load elite data and distribution
        elites_data_path = latest_timestamp_dir / cm.DATA_ELITES_FILE_NAME
        elites_summary_path = latest_timestamp_dir / cm.SUMMARY_ELITES_FILE_NAME

        if not elites_summary_path.exists():
            raise FileNotFoundError(f"Elite summary not found: {elites_summary_path}")

        # WARNING! Load distribution from HDF5 file
        try:
            with h5py.File(elites_summary_path, "r") as f:
                # Get most recent epoch
                epochs = sorted([int(k) for k in f.keys()])
                if not epochs:
                    raise ValueError("No epochs found in summary file")

                latest_epoch = epochs[-1]
                distribution_data = f[str(latest_epoch)]["elites"]["distribution"][:]

                logger.info(f"Loaded distribution from epoch {latest_epoch}")
        except Exception as e:
            logger.error(f"Failed to load distribution: {e}")
            raise

        # Load elite data for max merit factor
        max_merit_dict = {}
        if elites_data_path.exists():
            try:
                with h5py.File(elites_data_path, "r") as f:
                    epochs = sorted([int(k) for k in f.keys()])
                    if epochs:
                        latest = str(epochs[-1])
                        bin_seqs = torch.from_numpy(
                            f[latest]["elites"]["bin_seq_2d"][:]
                        ).to(cm.T_INT32)
                        mfs = torch.from_numpy(
                            f[latest]["elites"]["merit_factors_1d"][:]
                        ).to(cm.T_FLOAT32)

                        # Find maximum merit factor
                        max_idx = torch.argmax(mfs).item()
                        max_mf = mfs[max_idx].item()

                        max_merit_dict = {
                            "bin_seq_2d": bin_seqs[max_idx:max_idx+1],
                            "merit_factor": max_mf
                        }

                        logger.info(f"Loaded max merit factor: {max_mf:.6f}")
            except Exception as e:
                logger.warning(f"Failed to load elite data: {e}")

        # Pad distribution to current length
        distribution_tensor = torch.from_numpy(distribution_data).to(cm.T_FLOAT32)

        # WARNING! Pad distribution to match current sequence length
        if len(distribution_tensor) < len_bin_seq:
            padding = torch.full(
                (len_bin_seq - len(distribution_tensor),),
                0.5,
                dtype=cm.T_FLOAT32
            )
            distribution_tensor = torch.cat([distribution_tensor, padding])
            logger.info(f"Padded distribution from {len(distribution_data)} to {len_bin_seq}")
        elif len(distribution_tensor) > len_bin_seq:
            distribution_tensor = distribution_tensor[:len_bin_seq]
            logger.info(f"Truncated distribution from {len(distribution_data)} to {len_bin_seq}")

        return distribution_tensor, max_merit_dict

    def initialize(
        self,
        len_bin_seq: int,
        method_dist: str,
        num_limbs: int
    ) -> CEMInitResult:
        """
        Initialize CEM with all necessary components.

        Parameters
        ----------
        len_bin_seq : int
            Length of binary sequences.
        method_dist : str
            Distribution method ("naive" or "recursive").
        num_limbs : int
            Number of limbs per sequence.

        Returns
        -------
        CEMInitResult
            Complete initialization result.

        Raises
        ------
        ValueError
            If configuration is invalid.

        WARNING! All parameters must be validated before processing
        WARNING! Directories must be created before logging

        Examples
        --------
        >>> initializer = CEMInitializer()
        >>> result = initializer.initialize(256, "naive", 8)
        >>> logger = result.logger
        """
        # Validate configuration
        try:
            config = CEMInitConfig(
                len_bin_seq=len_bin_seq,
                method_dist=method_dist,
                num_limbs=num_limbs
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid CEM configuration: {e}")

        # Create directories and logger
        timestamp_log_dir, method_len_dir = self._create_log_directory(
            method_dist,
            len_bin_seq
        )

        log_file_path = Path(timestamp_log_dir) / cm.LOG_FILE_NAME
        logger = setup_logger(str(log_file_path))

        logger.info("=" * 80)
        logger.info("Cross-Entropy Method for Golay Merit Factor Problem")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config}")
        logger.info(f"Log directory: {timestamp_log_dir}")

        # Initialize distribution
        try:
            if method_dist == "naive":
                distribution, max_merit_dict = self._initialize_naive_distribution(
                    len_bin_seq,
                    logger
                )
            elif method_dist == "recursive":
                distribution, max_merit_dict = self._initialize_recursive_distribution(
                    method_len_dir,
                    len_bin_seq,
                    logger
                )
            else:
                raise ValueError(f"Unknown method: {method_dist}")

        except Exception as e:
            logger.error(f"Failed to initialize distribution: {e}", exc_info=True)
            raise

        # Create file paths
        summary_bin_seq_path = str(Path(method_len_dir) / cm.SUMMARY_BIN_SEQ_FILE_NAME)
        summary_elites_path = str(Path(method_len_dir) / cm.SUMMARY_ELITES_FILE_NAME)
        data_elites_path = str(Path(method_len_dir) / cm.DATA_ELITES_FILE_NAME)

        logger.info(f"Summary bin seq path: {summary_bin_seq_path}")
        logger.info(f"Summary elites path: {summary_elites_path}")
        logger.info(f"Data elites path: {data_elites_path}")

        return CEMInitResult(
            distribution=distribution,
            log_dir=timestamp_log_dir,
            summary_bin_seq_path=summary_bin_seq_path,
            summary_elites_path=summary_elites_path,
            data_elites_path=data_elites_path,
            max_merit_dict=max_merit_dict,
            logger=logger
        )


# ============================================================================
# Convenience Function (Legacy API)
# ============================================================================

def initialize_cem(
    len_bin_seq: int,
    method_dist: str,
    num_limbs: int,
    base_dir: str = "."
) -> CEMInitResult:
    """
    Convenience function to initialize CEM (legacy API).

    Parameters
    ----------
    len_bin_seq : int
        Length of binary sequences.
    method_dist : str
        Distribution method ("naive" or "recursive").
    num_limbs : int
        Number of limbs per sequence.
    base_dir : str, optional
        Base directory for logs. Default is current directory.

    Returns
    -------
    CEMInitResult
        Complete initialization result.

    Examples
    --------
    >>> result = initialize_cem(256, "naive", 8)
    >>> logger = result.logger
    """
    initializer = CEMInitializer(base_dir=base_dir)
    return initializer.initialize(len_bin_seq, method_dist, num_limbs)

if __name__ == "__main__":
    """Comprehensive example demonstrating CEM initialization."""

    print("\n" + "=" * 80)
    print("CEM Initializer - Example Usage")
    print("=" * 80)

    try:
        # Example 1: Naive initialization
        print("\n[Example 1] Naive Distribution Initialization")
        print("-" * 80)

        result = initialize_cem(
            len_bin_seq=256,
            method_dist="naive",
            num_limbs=8,
            base_dir="./example_cem"
        )

        logger = result.logger
        logger.info(f"Distribution shape: {result.distribution.shape}")
        logger.info(f"Distribution mean: {result.distribution.mean():.6f}")
        logger.info(f"Max merit dict: {result.max_merit_dict}")

        print(f"  Initialized CEM with naive method")
        print(f"  Log dir: {result.log_dir}")
        print(f"  Distribution shape: {result.distribution.shape}")

        # Example 2: Different sequence lengths
        print("\n[Example 2] Multiple Sequence Lengths")
        print("-" * 80)

        for len_seq in [64, 128, 256]:
            num_limbs = (len_seq + 31) // 32
            result = initialize_cem(
                len_bin_seq=len_seq,
                method_dist="naive",
                num_limbs=num_limbs,
                base_dir="./example_cem"
            )
            print(f" len_bin_seq={len_seq}, num_limbs={num_limbs}")

        # Example 3: Error handling
        print("\n[Example 3] Error Handling")
        print("-" * 80)

        error_cases = [
            {"len": 10, "method": "naive", "should_fail": True},  # Too small
            {"len": 5000, "method": "naive", "should_fail": True},  # Too large
            {"len": 256, "method": "invalid", "should_fail": True},  # Invalid method
            {"len": 256, "method": "naive", "should_fail": False},  # Valid
        ]

        for case in error_cases:
            try:
                num_limbs = (case["len"] + 31) // 32
                result = initialize_cem(
                    len_bin_seq=case["len"],
                    method_dist=case["method"],
                    num_limbs=num_limbs,
                    base_dir="./example_cem"
                )
                if case["should_fail"]:
                    print(f"   Expected failure but passed")
                else:
                    print(f"   len={case['len']}, method={case['method']}: Passed")
            except (ValueError, TypeError, FileNotFoundError) as e:
                if case["should_fail"]:
                    print(f"   len={case['len']}, method={case['method']}: Failed as expected")
                else:
                    print(f"   Unexpected error: {e}")

        print("\n" + "=" * 80)
        print(" All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n Error during example: {e}\n")
        raise
