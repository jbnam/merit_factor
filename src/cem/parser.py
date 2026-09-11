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

Author: NAMARI
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
from email import parser
import logging
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

import src.common as cm

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method parameters.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment (used in logging and file paths).
    len_bin_seq : int
        Bit length of binary sequences (16-4096 bits).
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
    experiment_name: str
    len_bin_seq: int
    method_dist: str
    num_epochs: int
    num_samples: int
    ratio: float
    log_level: int = logging.INFO

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_all()

    def _validate_all(self) -> None:
        """Validate all configuration parameters."""
        self._validate_experiment_name()
        self._validate_len_bin_seq()
        self._validate_method_dist()
        self._validate_num_epochs()
        self._validate_num_samples()
        self._validate_ratio()
        self._validate_log_level()

    def _validate_experiment_name(self) -> None:
        """Validate experiment name."""
        if not isinstance(self.experiment_name, str):
            logger.error(f"experiment_name must be a string: {type(self.experiment_name)}")
            raise TypeError(f"experiment_name must be a string: {type(self.experiment_name)}")

        if not self.experiment_name.strip():
            logger.error("experiment_name cannot be empty or whitespace.")
            raise ValueError("experiment_name cannot be empty or whitespace.")

    def _validate_len_bin_seq(self) -> None:
        """Validate binary sequence length."""
        if not isinstance(self.len_bin_seq, int):
            logger.error(f"len_bin_seq must be int: {type(self.len_bin_seq)}")
            raise TypeError(f"len_bin_seq must be int: {type(self.len_bin_seq)}")

        if self.len_bin_seq < cm.MIN_BIT_LEN:
            logger.error(f"len_bin_seq {self.len_bin_seq} is less than the minimum length allowed {cm.MIN_BIT_LEN}.")
            raise ValueError(f"len_bin_seq {self.len_bin_seq} is less than the minimum length allowed {cm.MIN_BIT_LEN}.")

        if self.len_bin_seq > cm.MAX_BIT_LEN:
            logger.error(f"len_bin_seq {self.len_bin_seq} exceeds the maximum length allowed {cm.MAX_BIT_LEN}.")
            raise ValueError(f"len_bin_seq {self.len_bin_seq} exceeds the maximum length allowed {cm.MAX_BIT_LEN}.")

    def _validate_method_dist(self) -> None:
        """Validate distribution method."""

        if self.method_dist not in cm.DIST_METHODS:
            logger.error(f"Invalid method_dist: {self.method_dist} Valid options: {cm.DIST_METHODS}")
            raise ValueError(f"method_dist must be one of {cm.DIST_METHODS}: '{self.method_dist}'")

    def _validate_num_epochs(self) -> None:
        """Validate number of epochs."""
        if not isinstance(self.num_epochs, int):
            logger.error(f"num_epochs must be an integer: {type(self.num_epochs)}")
            raise TypeError(f"num_epochs must be an integer: {type(self.num_epochs)}")

        if self.num_epochs <= 0:
            logger.error(f"num_epochs {self.num_epochs} is not positive.")
            raise ValueError(f"num_epochs must be positive: {self.num_epochs}")

    def _validate_num_samples(self) -> None:
        """Validate number of samples."""
        if not isinstance(self.num_samples, int):
            logger.error(f"num_samples must be an integer: {type(self.num_samples)}")
            raise TypeError(f"num_samples must be an integer: {type(self.num_samples)}")

        if self.num_samples <= 0:
            logger.error(f"num_samples {self.num_samples} is not positive.")
            raise ValueError(f"num_samples must be positive: {self.num_samples}")

        if self.num_samples > cm.MAX_SAMPLE_SIZE:
            logger.error(f"num_samples {self.num_samples} exceeds the maximum allowed {cm.MAX_SAMPLE_SIZE}.")
            raise ValueError(f"num_samples must not exceed {cm.MAX_SAMPLE_SIZE}: {self.num_samples}")

    def _validate_ratio(self) -> None:
        """Validate elite selection ratio."""
        if not isinstance(self.ratio, float):
            try:
                self.ratio = float(self.ratio)
            except (ValueError, TypeError) as e:
                logger.error(f"ratio must be convertible to float: {self.ratio}: {e}")
                raise TypeError(f"ratio must be convertible to float: {self.ratio}: {e}")

        if self.ratio <= 0.0 or self.ratio > 1.0:
            logger.error(f"ratio {self.ratio} is not in range (0.0, 1.0].")
            raise ValueError(f"ratio must be in range (0.0, 1.0]: {self.ratio}")

    def _validate_log_level(self) -> None:
        """Validate logging level."""
        if not isinstance(self.log_level, int):
            logger.error(f"log_level must be an integer: {type(self.log_level)}")
            raise TypeError(f"log_level must be an integer: {type(self.log_level)}")

        valid_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        if self.log_level not in valid_levels:
            logger.error(f"log_level {self.log_level} is not a valid logging level. Valid levels: {valid_levels}")
            raise ValueError(f"log_level must be one of {valid_levels}: {self.log_level}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CEMConfig(experiment_name='{self.experiment_name}', "
            f"len_bin_seq={self.len_bin_seq}, "
            f"method_dist='{self.method_dist}', num_epochs={self.num_epochs}, "
            f"num_samples={self.num_samples}, ratio={self.ratio}, "
            f"log_level={self.log_level})"
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

    Examples
    --------
    >>> parser = CEMArgumentParser()
    >>> config = parser.parse_args(["--len_bin_seq", "256"])
    >>> print(config)
    """
    def __init__(
        self,
        description: str = "Cross-Entropy Method for Golay's Merit Factor Problem"
    ):
        """Initialize argument parser."""
        self.description = description
        self.parser = self._create_parser()
        logger.info("======== CEMArgumentParser initialized ======================")

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

        # Arguments
        parser.add_argument(
            "--len_bin_seq",
            type=int,
            required=True,
            help="Length of binary sequences in bits in [16, 4096]"
        )

        parser.add_argument(
            "--experiment_name",
            type=str,
            required=True,
            help="Name of the experiment as string (used in logging and file paths)"
        )

        parser.add_argument(
            "--method_dist",
            type=str,
            default=cm.DIST_METHODS[0],
            choices=cm.DIST_METHODS,
            help=f"Method for generating Bernoulli distributions: {cm.DIST_METHODS}. "
                 f"Default: {cm.DIST_METHODS[0]}"
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
            default=6000,
            help="Number of samples per epoch. Default: 6000"
        )

        parser.add_argument(
            "--ratio",
            type=float,
            default=0.1,
            help="Elite selection ratio (0.0 to 1.0]. Default: 0.1"
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
            version="cem_Golay_merit_factor v0.1"
        )

        return parser

    def _get_epilog(self) -> str:
        """Get epilog with usage examples."""
        return """
Examples:
  # Basic usage with default parameters
  python golay.py --len_bin_seq 256 --experiment_name test_1

  # Custom configuration
  python golay.py --len_bin_seq 512 --experiment_name test_1 --num_epochs 20 --num_samples 500 --ratio 0.2

  # Using recursive method for distribution
  python golay.py --len_bin_seq 256 --experiment_name test_1 --method_dist recursive --num_epochs 50

  # With debug logging
  python golay.py --len_bin_seq 256 --experiment_name test_1 --log_level DEBUG --log_dir ./debug_logs
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

            logger.info(f"Parsed arguments: {parsed_args}")

            # Set logging level
            log_level = getattr(logging, parsed_args.log_level)
            logger.setLevel(log_level)

            # Create and validate configuration
            config = CEMConfig(
                len_bin_seq=parsed_args.len_bin_seq,
                experiment_name=parsed_args.experiment_name,
                method_dist=parsed_args.method_dist,
                num_epochs=parsed_args.num_epochs,
                num_samples=parsed_args.num_samples,
                ratio=parsed_args.ratio,
                log_level=logging.getLevelName(parsed_args.log_level)
            )

            logger.info(f"Configuration loaded successfully: {config}")

            return config

        except (ValueError, TypeError) as e:
            logger.error(f"Configuration validation failed: {e}")
            print(f"\n Error: {e}\n")
            sys.exit(1)

        except SystemExit as e:
            # Re-raise SystemExit (from argparse help/error)
            raise

        except Exception as e:
            logger.exception(f"Unexpected error during parsing: {e}")
            print(f"\n Unexpected error: {e}\n")
            sys.exit(1)

    def print_help(self) -> None:
        """Print help message."""
        self.parser.print_help()


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_args(
    args: Optional[List[str]] = None,
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
    """Comprehensive example demonstrating argument parsing."""

    print("\n" + "=" * 80)
    print("CEM Golay Merit Factor - Argument Parser Examples")
    print("=" * 80)

    try:
        # Example 1: Basic parsing with defaults
        print("\n[Example 1] Basic Parsing with Required Argument")
        print("-" * 80)

        config1 = parse_args(["--len_bin_seq", "256", "--experiment_name", "test_1"])
        print(f" Config 1: {config1}")

        # Example 2: Custom configuration
        print("\n[Example 2] Custom Configuration")
        print("-" * 80)

        config2 = parse_args([
            "--len_bin_seq", "512",
            "--experiment_name", "test_2",
            "--method_dist", "recursive",
            "--num_epochs", "20",
            "--num_samples", "500",
            "--ratio", "0.25"
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
                "--experiment_name", "test_4",
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
                "--experiment_name", "test_",
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

        # Example 7: Logging levels
        print("\n[Example 7] Different Logging Levels")
        print("-" * 80)

        for level in ["DEBUG", "INFO", "WARNING"]:
            config = parse_args([
                "--len_bin_seq", "256",
                "--experiment_name", "test_7",
                "--log_level", level
            ])
            print(f" Config with {level} logging level created")

        # Example 8: Parameter validation edge cases
        print("\n[Example 8] Parameter Validation Edge Cases")
        print("-" * 80)

        test_cases = [
            {
                "name": "Minimum valid ratio",
                "args": ["--len_bin_seq", "256", "--experiment_name", "test_8", "--ratio", "0.001"],
                "should_pass": True
            },
            {
                "name": "Maximum valid ratio",
                "args": ["--len_bin_seq", "256", "--experiment_name", "test_8", "--ratio", "1.0"],
                "should_pass": True
            },
            {
                "name": "Zero ratio (invalid)",
                "args": ["--len_bin_seq", "256", "--experiment_name", "test_8", "--ratio", "0.0"],
                "should_pass": False
            },
            {
                "name": "Small sequence length",
                "args": ["--len_bin_seq", "1", "--experiment_name", "test_8"],
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


    # Uncomment to parse actual command-line arguments
    # config = parse_args()
    # print(f"Configuration: {config}")
