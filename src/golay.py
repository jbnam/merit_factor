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

Author: NAMRI
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

logger = logging.getLogger(__name__)

# ============================================================================
# Enumerations
# ============================================================================

class DistributionMethod(Enum):
    """Enumeration for distribution generation methods.

    Attributes
    ----------
    NAIVE : str
        Naive method for generating Bernoulli distributions.
    RECURSIVE : str
        Recursive method for generating distributions recursively.
    """
    NAIVE = cm.METHOD_DIST_OPTIONS[0]     # naive
    RECURSIVE = cm.METHOD_DIST_OPTIONS[1] # recursive

    @classmethod
    def valid_methods(cls) -> List[str]:
        """Get list of valid method names."""
        return [method.value for method in cls]


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method parameters.

    Attributes
    ----------
    len_bin_seq : int
        Length of binary sequences (bits).
    method_dist : str
        Distribution generation method ("naive" or "recursive").
    num_epochs : int
        Number of training epochs.
    num_samples : int
        Number of samples per epoch.
    ratio : float
        Elite selection ratio (0.0 to 1.0].

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    len_bin_seq: int
    method_dist: str
    num_epochs: int
    num_samples: int
    ratio: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_all()

    def _validate_all(self) -> None:
        """Validate all configuration parameters."""
        self._validate_len_bin_seq()
        self._validate_method_dist()
        self._validate_num_epochs()
        self._validate_num_samples()
        self._validate_ratio()

    def _validate_len_bin_seq(self) -> None:
        """Validate binary sequence length."""
        if not isinstance(self.len_bin_seq, int):
            raise TypeError(f"len_bin_seq must be int, got {type(self.len_bin_seq)}")

        if self.len_bin_seq < cm.MIN_BIT_LEN:
            raise ValueError(f"len_bin_seq must be at least {cm.MIN_BIT_LEN}, got {self.len_bin_seq}")

        if self.len_bin_seq > cm.MAX_BIT_LEN:
            raise ValueError(f"len_bin_seq must be at most {cm.MAX_BIT_LEN}, got {self.len_bin_seq}")

    def _validate_method_dist(self) -> None:
        """Validate distribution method."""
        valid_methods = DistributionMethod.valid_methods()

        if self.method_dist not in valid_methods:
            raise ValueError(
                f"method_dist must be one of {valid_methods}, "
                f"got '{self.method_dist}'"
            )

    def _validate_num_epochs(self) -> None:
        """Validate number of epochs."""
        if not isinstance(self.num_epochs, int):
            raise TypeError(f"num_epochs must be int, got {type(self.num_epochs)}")

        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.num_epochs > cm.MAX_NUM_EPOCHS:
            raise ValueError(f"num_epochs must be at most {cm.MAX_NUM_EPOCHS}, got {self.num_epochs}")

    def _validate_num_samples(self) -> None:
        """Validate number of samples."""
        if not isinstance(self.num_samples, int):
            raise TypeError(f"num_samples must be int, got {type(self.num_samples)}")

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.num_samples > cm.MAX_SAMPLE_SIZE:
            raise ValueError(f"num_samples must be at most {cm.MAX_SAMPLE_SIZE}, got {self.num_samples}")

    def _validate_ratio(self) -> None:
        """Validate elite selection ratio."""
        if not isinstance(self.ratio, float):
            try:
                self.ratio = float(self.ratio)
            except (ValueError, TypeError) as e:
                raise TypeError(f"ratio must be convertible to float, got {self.ratio}: {e}")

        if self.ratio <= 0.0 or self.ratio > 1.0:
            raise ValueError(
                f"ratio must be in range (0.0, 1.0], got {self.ratio}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CEMConfig(len_bin_seq={self.len_bin_seq}, "
            f"method_dist='{self.method_dist}', num_epochs={self.num_epochs}, "
            f"num_samples={self.num_samples}, ratio={self.ratio} "
        )


# ============================================================================
# Argument Parser
# ============================================================================

class CEMArgumentParser:
    """Argument parser for CEM Golay Merit Factor Problem.

    Parameters
    ----------
    description : str, optional
        Parser description.

    Examples
    --------
    >>> parser = CEMArgumentParser()
    >>> config = parser.parse_args(["--len_bin_seq", "256"])
    >>> print(config)
    """

    def __init__(
        self,
        description: str = "Two-Way Cross-Entropy Method for Golay's Merit Factor Problem"
    ):
        """Initialize argument parser."""
        self.description = description
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser.

        Returns
        -------
        argparse.ArgumentParser
            Configured argument parser.
        """
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Required arguments
        parser.add_argument(
            "--len_bin_seq",
            type=int,
            required=True,
            help="Length of binary sequences in bits in [16, 4096]"
        )

        # Optional arguments with defaults
        parser.add_argument(
            "--method_dist",
            type=str,
            default=DistributionMethod.NAIVE.value,
            choices=DistributionMethod.valid_methods(),
            help=f"Method for generating Bernoulli distributions"
                 f"Choices: {DistributionMethod.valid_methods()}"
                 f"Default: {DistributionMethod.NAIVE.value}"
        )

        parser.add_argument(
            "--num_epochs",
            type=int,
            default=10,
            help="Number of training epochs. Default: 10"
        )

        parser.add_argument(
            "--num_samples",
            type=int,
            default=1000,
            help="Number of samples per epoch. Default: 100"
        )

        parser.add_argument(
            "--ratio",
            type=float,
            default=0.5,
            help="Elite selection ratio (0.0 to 1.0]. Default: 0.5"
        )

        parser.add_argument(
            "--log_dir",
            type=str,
            default="./logs",
            help="Directory for log files. Default: ./logs"
        )

        parser.add_argument(
            "-v", "--version",
            action="version",
            version="cem_Golay_merit_factor v0.2"
        )

        return parser

    def parse_args(
        self,
        args: Optional[List[str]] = None
    ) -> CEMConfig:
        """Parse and validate command-line arguments.

        Parameters
        ----------
        args : list of str, optional
            Arguments to parse. If None, uses sys.argv[1:].

        Returns
        -------
        CEMConfig
            Validated configuration object.

        Raises
        ------
        ValueError
            If any argument is invalid.
        SystemExit
            If parsing fails or help is requested.

        Examples
        --------
        >>> parser = CEMArgumentParser()
        >>> config = parser.parse_args(["--len_bin_seq", "256"])
        >>> print(f"Loaded config: {config}")
        """
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)

            logger.info(f"Parsed arguments: {parsed_args}")

            # Create and validate configuration
            config = CEMConfig(
                len_bin_seq=parsed_args.len_bin_seq,
                method_dist=parsed_args.method_dist,
                num_epochs=parsed_args.num_epochs,
                num_samples=parsed_args.num_samples,
                ratio=parsed_args.ratio
            )

            logger.info(f"Configuration loaded successfully: {config}")

            return config

        except (ValueError, TypeError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise RuntimeError(f"\nError: {e}\n")
            sys.exit(1)

        except SystemExit as e:
            # Re-raise SystemExit (from argparse help/error)
            raise

        except Exception as e:
            logger.exception(f"Unexpected error during parsing: {e}")
            raise RuntimeError(f"\nUnexpected error: {e}\n")

    def print_help(self) -> None:
        """Print help message."""
        self.parser.print_help()


# ============================================================================
# Convenience Function
# ============================================================================

def parse_args(
    args: Optional[List[str]]
) -> CEMConfig:
    """Convenience function to parse arguments.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse. Default uses sys.argv[1:].

    Returns
    -------
    CEMConfig
        Validated configuration.

    Examples
    --------
    >>> config = parse_args()
    >>> print(config.len_bin_seq)
    """
    parser = CEMArgumentParser()
    return parser.parse_args(args)

if __name__ == "__main__":
  # Set GPU environment
  if torch.cuda.is_available():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1"

  config = parse_args()
  print(f"Configuration: {config}")
