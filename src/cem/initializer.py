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

Author: MJMARI
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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import h5py

import src.common as cm
import src.cem.parser as parser
import src.utils.logger as mf_logger
import src.utils.cuda_parameters as cuda_params
import src.utils.stats as stats

# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class CEMInitConfig:
    """Configuration for CEM initialization.

    Attributes
    ----------
    experiment_name : str
        Name of an experiment (default: "cem").
    len_bin_seq : int
        Length of binary sequences (16-4096).
    method_dist : str
        Distribution method ("naive" or "recursive").
    num_epochs : int
        Number of epochs
    num_samples : int
        Number of binary sequences to be chosen.
    ratio : float
        Ratio of the number of elites to number of binary sequeneces per iteration.
    num_limbs : int
        Number of limbs per sequence.
    grid_size : int
        Number of block per grid.
    block_size : int
        Number of threads per block.
    num_iterations : int
        Number of iterations per epoch.
    """
    experiment_name: str = "cem"
    len_bin_seq: int
    method_dist: str
    num_epochs: int
    num_samples: int
    ratio: float
    num_elites_saved: int
    num_limbs: int
    grid_size: int
    block_size: int
    num_iterations: int

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
    logger : mf_logger.ExperimentLogger
        Configured logger instance.
    """
    distribution: torch.Tensor

    log_dir: str
    timestamp_log_dir: str
    method_len_dir: str
    summary_bin_seq_path: str
    summary_elites_path: str
    data_elites_path: str

    logger: mf_logger.ExperimentLogger


# ============================================================================
# CEM Initializer
# ============================================================================

class CEMInitializer:
    """Professional CEM initializer with robust configuration management.

    Parameters
    ----------
    pars_config : parser.CEMConfig
        CEM Configuration parsed.

    Examples
    --------
    >>> pars_config = parser.CEMConfig()
    >>> initializer = CEMInitializer(pars_config)
    >>> result = initializer.initialize()
    """

    def __init__(self, pars_config : parser.CEMConfig):
        """Initialize CEM initializer."""
        self.config = CEMInitConfig()

        # Set up mf_logger.ExperimentLogger()
        self.logger = mf_logger.ExperimentLogger(experiment_name=self.config.experiment_name)

        self.init_results = self.initisize(pars_config)

    def _create_log_directory(self) -> Tuple[str, str]:
        """
        Create log directory structure and return paths.

        Parameters
        ----------

        Returns
        -------
        tuple of (str, str)
            (timestamp_log_dir, method_len_dir)
        """
        # Create method/length directory structure
        method_dir = self.logger.log_dir / self.config.method_dist
        method_dir.mkdir(parents=True, exist_ok=True)

        method_len_dir = method_dir / str(self.config.len_bin_seq)
        method_len_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_log_dir = method_len_dir / timestamp
        timestamp_log_dir.mkdir(parents=True, exist_ok=True)

        return str(method_len_dir), str(timestamp_log_dir)

    def initialize(
        self,
        pars_config : parser.CEMConfig
    ) -> CEMInitResult:
        """
        Initialize CEM with all necessary components.

        Parameters
        ----------
        pars_config : parser.CEMConfig
            Parser configuration.

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
        >>> pars_config = pars.CEMConfig()
        >>> initializer = CEMInitializer(pars_config)
        >>> result = initializer.initialize(pars_config)
        >>> logger = result.logger
        """
        self.config.experiment_name = pars_config.experiment_name
        self.config.len_bin_seq = pars_config.len_bin_seq
        self.config.method_dist = pars_config.method_dist
        self.config.num_epochs = pars_config.num_epochs
        self.config.num_samples = pars_config.num_samples
        self.config.ratio = pars_config.ratio

        # Compute the num_limbs and num_elites_saved
        self.config.num_limbs = (self.config.len_bin_seq + cm.LIMB_BIT_SIZE - 1) // cm.LIMB_BIT_SIZE

        # Compute num_elites_saved,
        gpu_params = cuda_params.GPUArchitectureParams()
        dim_calculator = cuda_params.KernelDimensionCalculator(gpu_params)
        dimensions = dim_calculator.compute_dimensions(self.config.num_samples, self.config.num_limbs)

        self.config.grid_size = dimensions.grid_size
        self.config.block_size = dimensions.block_size
        self.config.num_iterations = dimensions.num_iterations

        # Create directories and logger
        method_len_dir, timestamp_log_dir = self._create_log_directory(
            self.config.method_dist,
            self.config.len_bin_seq
        )

        log_file_path = Path(timestamp_log_dir) / cm.LOG_FILE_NAME

        logger.info("=" * 80)
        logger.info("Cross-Entropy Method for Golay Merit Factor Problem")
        logger.info("=" * 80)
        logger.info(f"Configuration: {self.config}")

        # Initialize distribution
        try:
            if self.config.method_dist == cm.METHOD_DIST[0]:
                distribution, max_merit_dict = self._initialize_naive_distribution(method_len_dir, timestamp_log_dir)
            elif self.config.method_dist == cm.METHOD_DIST[1]:
                distribution, max_merit_dict = self._initialize_recursive_distribution(method_len_dir, timestamp_log_dir)
            else:
                raise ValueError(f"Unknown method: {self.config.method_dist}")

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

    def _initialize_naive_distribution(self,
                                       method_len_dir: str,
                                       timestamp_log_dir: str) -> torch.Tensor:
        """
        Initialize naive distribution (uniform 0.5 probability).

        Parameters
        ----------
        method_len_dir
            Path of method_dist / len_bin_seq.

        timestamp_log_dir
            Path of method_len_dir / timestamp.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distribution, max_merit_dict)

        Examples
        --------
        >>>
        >>> dist, max_dict = initializer._initialize_naive_distribution()
        """

        # TODO TODO
        # Check if there exists a previously completed
        if

        logger.info("Initializing naive distribution (uniform 0.5)")

        distribution = torch.full(
            (self.config.len_bin_seq,),
            0.5,
            dtype=cm.T_FLOAT32
        )

        logger.info(f"Initialized naive distribution: shape={distribution.shape}")

        return distribution

    def _initialize_recursive_distribution(
        self,
        method_len_dir: str,
        timestamp_log_dir: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Initialize recursive distribution from previous run.

        Parameters
        ----------
        method_len_dir
            Path of method_dist / len_bin_seq.

        timestamp_log_dir
            Path of method_len_dir / timestamp.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distribution, max_merit_dict)

        Raises
        ------
        ValueError
            If no previous run found.

        """
        logger.info(f"Initializing recursive distribution from {method_len_dir}")

        # WARNING! Scan for directories with numeric names (sequence lengths)
        method_dir = Path(method_len_dir)
        if not method_dir.exists():
            raise ValueError(f"Method directory not found: {method_len_dir}")

        # Find all numeric subdirectories (previous sequence lengths)
        numeric_dirs = []
        for item in method_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                numeric_dirs.append((int(item.name), item))

        if not numeric_dirs:
            raise ValueError(
                f"No previous runs found in {method_len_dir}. "
                f"Use 'naive' method instead."
            )

        # Find the maximum length that is less than len_bin_seq
        # WARNING! Must find largest len_bin_seq that is < current len_bin_seq
        valid_dirs = [(length, path) for length, path in numeric_dirs if length < self.config.len_bin_seq]

        if not valid_dirs:
            raise ValueError(
                f"No previous run with sequence length < {self.config.len_bin_seq}. "
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
        if len(distribution_tensor) < self.config.len_bin_seq:
            padding = torch.full(
                (self.config.len_bin_seq - len(distribution_tensor),),
                0.5,
                dtype=cm.T_FLOAT32
            )
            distribution_tensor = torch.cat([distribution_tensor, padding])
            logger.info(f"Padded distribution from {len(distribution_data)} to {self.config.len_bin_seq}")
        elif len(distribution_tensor) > self.config.len_bin_seq:
            distribution_tensor = distribution_tensor[:self.config.len_bin_seq]
            logger.info(f"Truncated distribution from {len(distribution_data)} to {self.config.len_bin_seq}")

        return distribution_tensor, max_merit_dict

# ============================================================================
# Convenience Function (Legacy API)
# ============================================================================

def initialize_cem(
    pars_config : parser.CEMConfig
) -> CEMInitResult:
    """
    Convenience function to initialize CEM (legacy API).

    Parameters
    ----------
    pars_config : parser.CEMConfig
        Dataclass of CEMConfig in parser containing CEM initial variables.

    Returns
    -------
    CEMInitResult
        Complete initialization result.

    Examples
    --------
    >>> pars_config = parser.CEMConfig()
    >>> result = initialize_cem(pars_config)
    >>> logger = result.logger
    """
    initializer = CEMInitializer(pars_config)
    return initializer.initialize(pars_config)

if __name__ == "__main__":
    """Comprehensive example demonstrating CEM initialization."""

    print("\n" + "=" * 80)
    print("CEM Initializer - Example Usage")
    print("=" * 80)

    pars_config = parser.CEMConfig()
    initializer = CEMInitializer(pars_config)

    try:
        # Example 1: Naive initialization
        print("\n[Example 1] Naive Distribution Initialization")
        print("-" * 80)

        result = initialize_cem(pars_config)

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
            initializer.config.len_bin_seq = len_seq
            initializer.config.num_limbs = (initializer.config.len_bin_seq + cm.LIMB_BIT_SIZE -1) // cm.LIMB_BIT_SIZE
            print(f" len_bin_seq={initializer.config.len_bin_seq}, num_limbs={initializer.config.num_limbs}")

        # Example 3: Error handling
        print("\n[Example 3] Error Handling")
        print("-" * 80)

        error_cases = [
            {"len": 10, "method": cm.METHOD_DIST[0], "should_fail": True},  # Too small
            {"len": 5000, "method": cm.METHOD_DIST[0], "should_fail": True},  # Too large
            {"len": 256, "method": cm.METHOD_DIST[2], "should_fail": True},  # Invalid method
            {"len": 256, "method": cm.METHOD_DIST[0], "should_fail": False},  # Valid
        ]

        for case in error_cases:
            try:
                initializer.config.len_bin_seq = case['len']
                initializer.config.num_limbs = (initializer.config.len_bin_seq + cm.LIMB_BIT_SIZE -1) // cm.LIMB_BIT_SIZE
                initializer.config.method_dist = case['method']

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
