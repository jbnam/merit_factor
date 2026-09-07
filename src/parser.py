"""
Command-Line Argument Parser Module
====================================

A professional-grade argument parser for the Cross-Entropy Method (CEM)
applied to Golay's Merit Factor Problem, with comprehensive validation,
logging, and configuration management.

This module provides:
- Argument parsing with type validation
- Configuration validation and error handling
- Professional logging for parsing operations
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

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

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
        Recursive method for generating distributions.
    """
    NAIVE = "naive"
    RECURSIVE = "recursive"

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
        Bit length of binary sequences (16-4096 bits).
    method_dist : str
        Distribution generation method ("naive" or "recursive").
    num_epochs : int
        Number of training epochs.
    num_samples : int
        Number of samples per epoch.
    ratio : float
        Elite selection ratio (0.0 to 1.0).
    num_elites_saved : int
        Number of best elites to save per epoch.

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
    num_elites_saved: int

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
        self._validate_num_elites_saved()
        self._validate_consistency()

    def _validate_len_bin_seq(self) -> None:
        """Validate binary sequence length."""
        if not isinstance(self.len_bin_seq, int):
            raise TypeError(f"len_bin_seq must be int, got {type(self.len_bin_seq)}")

        if self.len_bin_seq <= 0:
            raise ValueError(f"len_bin_seq must be positive, got {self.len_bin_seq}")

        if self.len_bin_seq > 10000:
            logger.warning(f"len_bin_seq={self.len_bin_seq} is very large")

    def _validate_method_dist(self) -> None:
        """Validate distribution method."""
        valid_methods = DistributionMethod.valid_methods()

        if self.method_dist not in valid_methods:
            logger.error(
                f"Invalid method_dist: {self.method_dist}. "
                f"Valid options: {valid_methods}"
            )
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

        if self.num_epochs > 10000:
            logger.warning(f"num_epochs={self.num_epochs} is very large")

    def _validate_num_samples(self) -> None:
        """Validate number of samples."""
        if not isinstance(self.num_samples, int):
            raise TypeError(f"num_samples must be int, got {type(self.num_samples)}")

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.num_samples > 10000000:
            logger.warning(f"num_samples={self.num_samples} is very large")

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

    def _validate_num_elites_saved(self) -> None:
        """Validate number of elites to save."""
        if not isinstance(self.num_elites_saved, int):
            raise TypeError(
                f"num_elites_saved must be int, got {type(self.num_elites_saved)}"
            )

        if self.num_elites_saved <= 0:
            raise ValueError(
                f"num_elites_saved must be positive, got {self.num_elites_saved}"
            )

    def _validate_consistency(self) -> None:
        """Validate consistency between parameters."""
        # Compute number of elites from ratio
        num_elites = int(self.ratio * self.num_samples)

        if self.num_elites_saved > num_elites:
            logger.warning(
                f"num_elites_saved ({self.num_elites_saved}) exceeds "
                f"number of elites selected ({num_elites}). "
                f"Clamping num_elites_saved to {num_elites}."
            )
            self.num_elites_saved = num_elites

        if self.num_elites_saved > self.num_samples:
            raise ValueError(
                f"num_elites_saved ({self.num_elites_saved}) cannot exceed "
                f"num_samples ({self.num_samples})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CEMConfig(len_bin_seq={self.len_bin_seq}, "
            f"method_dist='{self.method_dist}', num_epochs={self.num_epochs}, "
            f"num_samples={self.num_samples}, ratio={self.ratio}, "
            f"num_elites_saved={self.num_elites_saved})"
        )


# ============================================================================
# Argument Parser
# ============================================================================

class CEMArgumentParser:
    """Professional argument parser for CEM Golay Merit Factor Problem.

    Parameters
    ----------
    description : str, optional
        Parser description.
    log_level : int, optional
        Logging level. Default is INFO.

    Examples
    --------
    >>> parser = CEMArgumentParser()
    >>> config = parser.parse_args(["--len_bin_seq", "256"])
    >>> print(config)
    """

    def __init__(
        self,
        description: str = "Cross-Entropy Method for Golay's Merit Factor Problem",
        log_level: int = logging.INFO
    ):
        """Initialize argument parser."""
        self.description = description
        self.logger = logger
        self.logger.setLevel(log_level)

        self.parser = self._create_parser()
        self.logger.info("CEMArgumentParser initialized")

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
            epilog=self._get_epilog()
        )

        # Required arguments
        parser.add_argument(
            "--len_bin_seq",
            type=int,
            required=True,
            help="Length of binary sequences in bits (e.g., 256, 512)"
        )

        # Optional arguments with defaults
        parser.add_argument(
            "--method_dist",
            type=str,
            default=DistributionMethod.NAIVE.value,
            choices=DistributionMethod.valid_methods(),
            help=f"Method for generating Bernoulli distributions. "
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
            default=100,
            help="Number of samples per epoch. Default: 100"
        )

        parser.add_argument(
            "--ratio",
            type=float,
            default=0.5,
            help="Elite selection ratio (0.0 to 1.0]. Default: 0.5"
        )

        parser.add_argument(
            "--num_elites_saved",
            type=int,
            default=10,
            help="Number of best elites to save per epoch. Default: 10"
        )

        # Logging and version
        parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level. Default: INFO"
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

    def _get_epilog(self) -> str:
        """Get epilog with usage examples."""
        return """
Examples:
  # Basic usage with default parameters
  python main.py --len_bin_seq 256

  # Custom configuration
  python main.py --len_bin_seq 512 --num_epochs 20 --num_samples 500 --ratio 0.2

  # Using recursive method for distribution
  python main.py --len_bin_seq 256 --method_dist recursive --num_epochs 50

  # With debug logging
  python main.py --len_bin_seq 256 --log_level DEBUG --log_dir ./debug_logs
"""

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

            self.logger.debug(f"Parsed arguments: {parsed_args}")

            # Configure logging level
            log_level = getattr(logging, parsed_args.log_level)
            setup_logging(parsed_args.log_dir, log_level)
            self.logger.setLevel(log_level)

            # Create and validate configuration
            config = CEMConfig(
                len_bin_seq=parsed_args.len_bin_seq,
                method_dist=parsed_args.method_dist,
                num_epochs=parsed_args.num_epochs,
                num_samples=parsed_args.num_samples,
                ratio=parsed_args.ratio,
                num_elites_saved=parsed_args.num_elites_saved
            )

            self.logger.info(f"Configuration loaded successfully: {config}")

            return config

        except (ValueError, TypeError) as e:
            self.logger.error(f"Configuration validation failed: {e}")
            print(f"\n❌ Error: {e}\n")
            sys.exit(1)

        except SystemExit as e:
            # Re-raise SystemExit (from argparse help/error)
            raise

        except Exception as e:
            self.logger.exception(f"Unexpected error during parsing: {e}")
            print(f"\n❌ Unexpected error: {e}\n")
            sys.exit(1)

    def print_help(self) -> None:
        """Print help message."""
        self.parser.print_help()


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_args(
    args: Optional[List[str]] = None,
    log_level: int = logging.INFO
) -> CEMConfig:
    """Convenience function to parse arguments.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse. Default uses sys.argv[1:].
    log_level : int, optional
        Logging level. Default is INFO.

    Returns
    -------
    CEMConfig
        Validated configuration.

    Examples
    --------
    >>> config = parse_args()
    >>> print(config.len_bin_seq)
    """
    parser = CEMArgumentParser(log_level=log_level)
    return parser.parse_args(args)


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Comprehensive example demonstrating argument parsing."""

    print("\n" + "=" * 80)
    print("CEM Golay Merit Factor - Argument Parser Examples")
    print("=" * 80)

    try:
        # Example 1: Basic parsing with defaults
        print("\n[Example 1] Basic Parsing with Required Argument")
        print("-" * 80)

        config1 = parse_args(["--len_bin_seq", "256"])
        print(f" Config 1: {config1}")

        # Example 2: Custom configuration
        print("\n[Example 2] Custom Configuration")
        print("-" * 80)

        config2 = parse_args([
            "--len_bin_seq", "512",
            "--method_dist", "recursive",
            "--num_epochs", "20",
            "--num_samples", "500",
            "--ratio", "0.25",
            "--num_elites_saved", "50"
        ])
        print(f" Config 2: {config2}")

        # Example 3: Configuration as dictionary
        print("\n[Example 3] Configuration as Dictionary")
        print("-" * 80)

        config_dict = config2.to_dict()
        print(f" Config as dict:")
        for key, value in config_dict.items():
            print(f"    {key}: {value}")

        # Example 4: Error handling - invalid ratio
        print("\n[Example 4] Error Handling - Invalid Ratio")
        print("-" * 80)

        try:
            config_bad = parse_args([
                "--len_bin_seq", "256",
                "--ratio", "1.5"
            ])
        except SystemExit:
            print(f" Invalid ratio properly rejected")

        # Example 5: Error handling - invalid method
        print("\n[Example 5] Error Handling - Invalid Method")
        print("-" * 80)

        try:
            config_bad = parse_args([
                "--len_bin_seq", "256",
                "--method_dist", "invalid"
            ])
        except SystemExit:
            print(f" Invalid method properly rejected")

        # Example 6: Error handling - missing required argument
        print("\n[Example 6] Error Handling - Missing Required Argument")
        print("-" * 80)

        try:
            config_bad = parse_args([])
        except SystemExit:
            print(f" Missing required argument properly detected")

        # Example 7: Error handling - inconsistent parameters
        print("\n[Example 7] Error Handling - Inconsistent Parameters")
        print("-" * 80)

        try:
            config_bad = parse_args([
                "--len_bin_seq", "256",
                "--num_samples", "100",
                "--num_elites_saved", "1000"  # More than samples
            ])
        except SystemExit:
            print(f" Inconsistent parameters properly detected")

        # Example 8: Logging levels
        print("\n[Example 8] Different Logging Levels")
        print("-" * 80)

        for level in ["DEBUG", "INFO", "WARNING"]:
            config = parse_args([
                "--len_bin_seq", "256",
                "--log_level", level
            ])
            print(f" Config with {level} logging level created")

        # Example 9: Parameter validation edge cases
        print("\n[Example 9] Parameter Validation Edge Cases")
        print("-" * 80)

        test_cases = [
            {
                "name": "Minimum valid ratio",
                "args": ["--len_bin_seq", "256", "--ratio", "0.001"],
                "should_pass": True
            },
            {
                "name": "Maximum valid ratio",
                "args": ["--len_bin_seq", "256", "--ratio", "1.0"],
                "should_pass": True
            },
            {
                "name": "Zero ratio (invalid)",
                "args": ["--len_bin_seq", "256", "--ratio", "0.0"],
                "should_pass": False
            },
            {
                "name": "Small sequence length",
                "args": ["--len_bin_seq", "1"],
                "should_pass": True
            },
        ]

        for test in test_cases:
            try:
                config = parse_args(test["args"])
                if test["should_pass"]:
                    print(f"   {test['name']}: Passed")
                else:
                    print(f"   {test['name']}: Expected failure but passed")
            except SystemExit:
                if not test["should_pass"]:
                    print(f"   {test['name']}: Failed as expected")
                else:
                    print(f"   {test['name']}: Unexpected failure")

        print("\n" + "=" * 80)
        print(" All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\n Error during example: {e}\n")


if __name__ == "__main__":
    # Uncomment to run examples
    example_usage()

    # Uncomment to parse actual command-line arguments
    # config = parse_args()
    # print(f"Configuration: {config}")
