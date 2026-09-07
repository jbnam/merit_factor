"""
Elite Binary Sequence Selector, Tensor Concatenation, and Reshaping Module
=============================================================================

A high-performance module for selecting elite binary sequences from a 2D tensor
based on their defined scores.

This module provides functionality to extract the top-performing binary sequences
(elite candidates) from a collection based on their associated scores,
with full support for CPU operations and performance optimization.

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Device Placement:
  - All operations are performed on CPU
  - Sign and magnitude handling follows Python/NumPy conventions

Performance Considerations:
  - torch.topk() is highly optimized for CPU tensors
  - Advanced indexing (fancy indexing) is used for efficient extraction

Sign and magnitude handling:
  - Sign and magnitude issues in Python/PyTorch CPU execution are handled explicitly
  - GPU bitwise operations maintain correct sign/magnitude semantics automatically
"""
from logging import getLogger
import time
from typing import Tuple, Optional
from dataclasses import dataclass

import torch

# get logger
logger = getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EliteSelectionConfig:
    """Configuration parameters for elite selection.

    Attributes
    ----------
    ratio : float
        Selection ratio (0.0 to 1.0).
    validate_input : bool
        Whether to perform comprehensive input validation. Default is True.
    preserve_indices : bool
        Whether to return original indices of selected sequences. Default is False.
    """
    ratio: float
    validate_input: bool = True
    preserve_indices: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.ratio <= 1.0):
            raise ValueError(f"Ratio must be in [0.0, 1.0], got {self.ratio}")


# ============================================================================
# Elite Selector Class
# ============================================================================

class EliteSelector:
    """High-performance elite binary sequence selector.

    This class provides functionality to extract top-performing binary sequences
    from a 2D tensor based on their associated scores. It includes comprehensive
    validation, performance profiling, and logging.

    Parameters
    ----------
    config : EliteSelectionConfig, optional
        Configuration parameters. If None, uses default config.

    Attributes
    ----------
    config : EliteSelectionConfig
        Configuration parameters.

    Examples
    --------
    >>> config = EliteSelectionConfig(ratio=0.5)
    >>> selector = EliteSelector(config=config)
    >>> bin_seq_2d = torch.randn(1000, 256)
    >>> mf_1d = torch.randn(1000)
    >>> elite_seq, elite_mf = selector.select_elites(bin_seq_2d, mf_1d)
    >>> print(f"Selected {elite_seq.shape[0]} elite sequences")
    """

    def __init__(self, config: Optional[EliteSelectionConfig] = None):
        """Initialize elite selector.

        Parameters
        ----------
        config : EliteSelectionConfig, optional
            Configuration parameters. Uses default if None.
        """
        self.config = config if config is not None else EliteSelectionConfig(ratio=0.5)

        logger.info(f"Initialized EliteSelector with ratio={self.config.ratio}")

    def _validate_tensor_device(self, tensor: torch.Tensor, name: str) -> None:
        """
        Validate that tensor is on CPU.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to validate.
        name : str
            Name of the tensor (for error messages).

        Raises
        ------
        TypeError
            If tensor is not on CPU.
        """

        if tensor.device.type != "cpu":
            logger.error(
                f"Device validation failed for {name}: "
                f"expected 'cpu', got '{tensor.device.type}'"
            )
            raise TypeError(
                f"The {name} does not reside on CPU. "
                f"Found: {tensor.device.type}"
            )

    def _validate_dimensions(
        self,
        bin_seq_2d: torch.Tensor,
        mf_1d: torch.Tensor
    ) -> None:
        """
        Validate tensor dimensions.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            2D binary sequences tensor.
        mf_1d : torch.Tensor
            1D merit factors tensor.

        Raises
        ------
        ValueError
            If dimensions are invalid.
        """
        if bin_seq_2d.dim() != 2:
            logger.error(
                f"Invalid dimension for bin_seq_2d: "
                f"expected 2D, got {bin_seq_2d.dim()}D with shape {bin_seq_2d.shape}"
            )
            raise ValueError(
                f"bin_seq_2d must be 2D tensor, got {bin_seq_2d.dim()}D "
                f"with shape {bin_seq_2d.shape}"
            )

        if mf_1d.dim() != 1:
            logger.error(
                f"Invalid dimension for mf_1d: "
                f"expected 1D, got {mf_1d.dim()}D with shape {mf_1d.shape}"
            )
            raise ValueError(
                f"mf_1d must be 1D tensor, got {mf_1d.dim()}D "
                f"with shape {mf_1d.shape}"
            )

    def _validate_shape_consistency(
        self,
        bin_seq_2d: torch.Tensor,
        mf_1d: torch.Tensor
    ) -> None:
        """
        Validate shape consistency between tensors.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            2D binary sequences tensor of shape (N, M).
        mf_1d : torch.Tensor
            1D merit factors tensor of shape (N,).

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """

        n_sequences = bin_seq_2d.shape[0]
        n_merit_factors = mf_1d.shape[0]

        if n_sequences != n_merit_factors:
            logger.error(
                f"Shape mismatch: bin_seq_2d has {n_sequences} sequences, "
                f"but mf_1d has {n_merit_factors} merit factors"
            )
            raise ValueError(
                f"First dimension mismatch: bin_seq_2d[0]={n_sequences} "
                f"vs mf_1d[0]={n_merit_factors}. "
                f"Each sequence must have a corresponding merit factor."
            )

    def _validate_ratio(self, ratio: float) -> None:
        """
        Validate selection ratio.

        Parameters
        ----------
        ratio : float
            Selection ratio.

        Raises
        ------
        ValueError
            If ratio is invalid.
        """

        if isinstance(ratio, torch.Tensor):
            logger.warning(f"Ratio is a tensor: {ratio}. Converting to float.")
            ratio = float(ratio.item())

        if not isinstance(ratio, (int, float)):
            logger.error(f"Invalid ratio type: {type(ratio)}")
            raise TypeError(f"ratio must be numeric, got {type(ratio)}")

        if not (0.0 <= ratio <= 1.0):
            logger.error(f"ratio out of range: {ratio}")
            raise ValueError(
                f"Ratio must be in range [0.0, 1.0] (inclusive), got {ratio}"
            )

    def _validate_num_elites(
        self,
        num_elites: int,
        total_sequences: int
    ) -> None:
        """
        Validate computed number of elite sequences.

        Parameters
        ----------
        num_elites : int
            Computed number of elite sequences.
        total_sequences : int
            Total number of sequences.

        Raises
        ------
        ValueError
            If num_elites is invalid.
        """

        if num_elites <= 0:
            logger.error(
                f"Invalid num_elites: {num_elites} "
                f"(computed from ratio={self.config.ratio} × total={total_sequences})"
            )
            raise ValueError(
                f"Number of elites must be > 0. Got {num_elites} "
                f"from ratio={self.config.ratio} × total={total_sequences}. "
                f"Consider increasing the ratio."
            )

        if num_elites > total_sequences:
            logger.error(f"num_elites ({num_elites}) exceeds total ({total_sequences})")
            raise ValueError(
                f"Number of elites ({num_elites}) cannot exceed total sequences ({total_sequences})"
            )

    def select_elites(
        self,
        bin_seq_2d: torch.Tensor,
        mf_1d: torch.Tensor,
        ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select elite binary sequences based on merit factor scores.

        Extracts the top ratio portion of binary sequences from bin_seq_2d
        based on the highest scores in mf_1d. All operations are performed
        on CPU.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            2D tensor of binary sequences with shape (N, M) where N is the
            number of sequences and M is the sequence length.
        mf_1d : torch.Tensor
            1D tensor of merit factors with shape (N,), where each element
            is the merit factor (score) for the corresponding sequence.
        ratio : float, optional
            Selection ratio in range [0.0, 1.0]. If None, uses config ratio.

        Returns
        -------
        tuple of torch.Tensor
            - elite_sequences: 2D tensor of selected elite sequences with shape
              (ceil(ratio × N), M), containing only the top-scoring sequences.
            - elite_merit_factors: 1D tensor of merit factors for selected
              sequences with shape (ceil(ratio × N),), in descending order.

        Raises
        ------
        TypeError
            If tensors are not on CPU or if ratio is invalid type.
        ValueError
            If tensor dimensions/shapes are invalid or ratio is out of range.

        Notes
        -----
        Sign and Magnitude Handling:
            - All operations use standard PyTorch semantics
            - Merit factor values are compared directly (no sign conversion needed)
            - Indexing preserves tensor properties (dtype, layout)

        Performance Notes:
            - torch.topk() is O(N log k) where k = num_elites
            - Tensor indexing is O(k) memory copy operation
            - Overall complexity: O(N log k)

        Examples
        --------
        >>> selector = EliteSelector(EliteSelectionConfig(ratio=0.5))
        >>> bin_seq_2d = torch.randn(1000, 256)
        >>> mf_1d = torch.randn(1000)
        >>> elite_seq, elite_mf = selector.select_elites(bin_seq_2d, mf_1d)
        >>> print(elite_seq.shape)  # Should be around (500, 256)
        >>> print(elite_mf.shape)   # Should be around (500,)

        >>> # With custom ratio
        >>> elite_seq, elite_mf = selector.select_elites(
        ...     bin_seq_2d, mf_1d, ratio=0.1
        ... )
        >>> print(elite_seq.shape)  # Should be around (100, 256)
        """
        logger.info(
            f"Starting elite selection: bin_seq_2d={bin_seq_2d.shape}, "
            f"mf_1d={mf_1d.shape}, ratio={ratio if ratio is not None else self.config.ratio}"
        )

        operation_start = time.time()

        # Use provided ratio or config ratio
        effective_ratio = ratio if ratio is not None else self.config.ratio

        # Input validation
        if self.config.validate_input:
            logger.debug("Performing input validation")

            self._validate_tensor_device(bin_seq_2d, "bin_seq_2d")
            self._validate_tensor_device(mf_1d, "mf_1d")
            self._validate_dimensions(bin_seq_2d, mf_1d)
            self._validate_shape_consistency(bin_seq_2d, mf_1d)
            self._validate_ratio(effective_ratio)

            logger.debug("Input validation completed successfully")

        # Get tensor dimensions
        n_sequences, m_sequence_length = bin_seq_2d.shape

        num_elites = int(effective_ratio * n_sequences)

        # Validate num_elites
        self._validate_num_elites(num_elites, n_sequences)

        logger.info(
            f"Selected configuration: n_sequences={n_sequences}, "
            f"sequence_length={m_sequence_length}, num_elites={num_elites}"
        )

        # Step 1: Find top-k merit factors and their indices
        logger.debug("Executing torch.topk() to find elite indices")
        topk_start = time.time()

        topk_mfs, topk_indices = torch.topk(mf_1d, k=num_elites, largest=True)

        topk_time = time.time() - topk_start
        logger.debug(f"torch.topk() completed in {topk_time:.6f}s")

        # Step 2: Index and extract elite sequences
        logger.debug("Extracting elite sequences using fancy indexing")
        indexing_start = time.time()

        elite_sequences = bin_seq_2d[topk_indices]

        indexing_time = time.time() - indexing_start
        logger.debug(f"Indexing completed in {indexing_time:.6f}s")

        # Ensure contiguity for downstream operations
        if not elite_sequences.is_contiguous():
            logger.debug("Making elite_sequences contiguous")
            elite_sequences = elite_sequences.contiguous()

        return elite_sequences, topk_mfs

    def select_elites_with_indices(
        self,
        bin_seq_2d: torch.Tensor,
        mf_1d: torch.Tensor,
        ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select elite binary sequences and return their original indices.

        This variant of select_elites() also returns the original indices
        of the selected sequences, useful for tracking which sequences were selected.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            2D tensor of binary sequences with shape (N, M).
        mf_1d : torch.Tensor
            1D tensor of merit factors with shape (N,).
        ratio : float, optional
            Selection ratio. Uses config ratio if None.

        Returns
        -------
        tuple of torch.Tensor
            - elite_sequences: Selected sequences with shape (ceil(ratio × N), M).
            - elite_merit_factors: Merit factors of selected sequences.
            - elite_indices: Original indices of selected sequences.

        Examples
        --------
        >>> selector = EliteSelector(EliteSelectionConfig(ratio=0.5))
        >>> bin_seq_2d = torch.randn(1000, 256)
        >>> mf_1d = torch.randn(1000)
        >>> elite_seq, elite_mf, indices = selector.select_elites_with_indices(
        ...     bin_seq_2d, mf_1d
        ... )
        >>> print(f"Selected indices: {indices}")
        """
        logger.info("Starting elite selection with index tracking")

        # Use provided ratio or config ratio
        effective_ratio = ratio if ratio is not None else self.config.ratio

        # Perform standard validation
        if self.config.validate_input:
            self._validate_tensor_device(bin_seq_2d, "bin_seq_2d")
            self._validate_tensor_device(mf_1d, "mf_1d")
            self._validate_dimensions(bin_seq_2d, mf_1d)
            self._validate_shape_consistency(bin_seq_2d, mf_1d)
            self._validate_ratio(effective_ratio)

        n_sequences = bin_seq_2d.shape[0]
        num_elites = int(effective_ratio * n_sequences)
        self._validate_num_elites(num_elites, n_sequences)

        topk_mfs, topk_indices = torch.topk(mf_1d, k=num_elites, largest=True)
        elite_sequences = bin_seq_2d[topk_indices]

        if not elite_sequences.is_contiguous():
            elite_sequences = elite_sequences.contiguous()

        logger.info(f"Selected {num_elites} sequences with indices")

        return elite_sequences, topk_mfs, topk_indices


# ============================================================================
# Convenience Function (Legacy API)
# ============================================================================

def find_elites(
    bin_seq_2d: torch.Tensor,
    mf_1d: torch.Tensor,
    ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the top ratio 1D tensor from bin_seq_2d based on highest scores in mf_1d.

    This is a convenience function that wraps the EliteSelector class for
    backward compatibility with the original API.

    Parameters
    ----------
    bin_seq_2d : torch.Tensor
        2D tensor of binary sequences with shape (N, M) where N is the number
        of sequences and M is the sequence length. Must reside on CPU.
    mf_1d : torch.Tensor
        1D tensor of merit factors with shape (N,), one for each sequence.
        Must reside on CPU.
    ratio : float
        Selection ratio in range [0.0, 1.0] (inclusive). The function will
        select approximately ratio * N top-scoring sequences.

    Returns
    -------
    tuple of torch.Tensor
        - elite_tensor: 2D tensor of selected elite sequences with shape
          (ceil(ratio * N), M), containing the sequences with highest merit factors.
        - topk_mfs: 1D tensor of merit factors for selected sequences with shape
          (ceil(ratio * N),), in descending order.

    Raises
    ------
    ValueError
        If tensor dimensions are invalid, shapes are inconsistent, or ratio
        is out of range.
    TypeError
        If tensors are not on CPU.

    Notes
    -----
    This function performs the following operations:

    1. Validates input tensor dimensions (bin_seq_2d must be 2D, mf_1d must be 1D)
    2. Validates shape consistency (both must have same first dimension)
    3. Validates ratio is in [0.0, 1.0]
    4. Ensures both tensors are on CPU
    5. Computes number of elite sequences: ceil(ratio × N)
    6. Uses torch.topk() to find top-scoring sequences
    7. Extracts elite sequences using fancy indexing

    Examples
    --------
    >>> bin_seq_2d = torch.randn(1000, 256)
    >>> mf_1d = torch.randn(1000)
    >>> elite_seq, elite_mf = find_elites(bin_seq_2d, mf_1d, ratio=0.5)
    >>> print(f"Selected {elite_seq.shape[0]} elite sequences")
    """
    # Use EliteSelector with default config
    selector = EliteSelector(
        config=EliteSelectionConfig(ratio=ratio, validate_input=True)
    )
    return selector.select_elites(bin_seq_2d, mf_1d, ratio=ratio)

if __name__ == "__main__":
    """Comprehensive example demonstrating elite selection usage."""

    print("\n" + "=" * 80)
    print("Elite Binary Sequence Selector - Example Usage")
    print("=" * 80)

    try:
        # Example 1: Basic usage with find_elites function
        print("\n[Example 1] Basic Usage with find_elites() Function")
        print("-" * 80)

        # Create sample data
        n_sequences = 1000
        sequence_length = 256

        bin_seq_2d = torch.randn(n_sequences, sequence_length, dtype=torch.float32)
        mf_1d = torch.randn(n_sequences, dtype=torch.float32)

        print(f"Input shapes:")
        print(f"  bin_seq_2d: {bin_seq_2d.shape}")
        print(f"  mf_1d:      {mf_1d.shape}")

        # Select top 50% of sequences
        ratio = 0.5
        elite_seq, elite_mf = find_elites(bin_seq_2d, mf_1d, ratio=ratio)

        print(f"\nOutput shapes (ratio={ratio}):")
        print(f"  elite_seq: {elite_seq.shape}")
        print(f"  elite_mf:  {elite_mf.shape}")

        # Verify merit factors are in descending order
        is_sorted = torch.all(elite_mf[:-1] >= elite_mf[1:])
        print(f"Merit factors in descending order: {is_sorted.item()}")

        # Example 2: Using select_elites_with_indices to track original positions
        print("\n[Example 2] Elite Selection with Original Indices")
        print("-" * 80)

        selector = EliteSelector(
                config=EliteSelectionConfig(ratio=ratio, validate_input=True)
            )

        elite_seq_idx, elite_mf_idx, original_indices = (
            selector.select_elites_with_indices(bin_seq_2d, mf_1d, ratio=0.1)
        )

        print(f"\nResults:")
        print(f"  Elite sequences shape:  {elite_seq_idx.shape}")
        print(f"  Original indices shape: {original_indices.shape}")
        print(f"  First 10 selected indices: {original_indices[:10].tolist()}")
        print(f"  Top 5 merit factors: {elite_mf_idx[:5].tolist()}")

        # Example 3: Different selection ratios
        print("\n[Example 3] Comparing Different Selection Ratios")
        print("-" * 80)

        ratios = [0.1, 0.25, 0.5, 0.75, 1.0]

        print(f"\n{'Ratio':<10} {'Selected':<15} {'Output Shape':<20} {'Memory Saved (MB)':<20}")
        print("-" * 65)

        for r in ratios:
            elite_seq_r, elite_mf_r = selector.select_elites(bin_seq_2d, mf_1d, ratio=r)

        # Example 4: Error handling demonstration
        print("\n[Example 4] Error Handling")
        print("-" * 80)

        test_cases = [
            {
                "name": "Invalid ratio (> 1.0)",
                "ratio": 1.5,
                "should_fail": True
            },
            {
                "name": "Invalid ratio (< 0.0)",
                "ratio": -0.5,
                "should_fail": True
            },
            {
                "name": "Valid ratio (0.0)",
                "ratio": 0.0,
                "should_fail": True  # num_elites would be 0
            },
            {
                "name": "Valid ratio (1.0)",
                "ratio": 1.0,
                "should_fail": False
            },
        ]

        for test_case in test_cases:
            try:
                elite_seq_test, elite_mf_test = selector.select_elites(
                    bin_seq_2d, mf_1d, ratio=test_case["ratio"]
                )
                if test_case["should_fail"]:
                    print(f"   {test_case['name']}: Expected failure but succeeded")
                else:
                    print(f"   {test_case['name']}: Successfully selected {elite_seq_test.shape[0]} sequences")
            except (ValueError, TypeError) as e:
                if test_case["should_fail"]:
                    print(f"   {test_case['name']}: Failed as expected")
                else:
                    print(f"   {test_case['name']}: Unexpected failure - {e}")

        # Example 5: Large-scale test
        print("\n[Example 5] Large-Scale Performance Test")
        print("-" * 80)

        large_n = 100000
        large_m = 512

        print(f"Creating large tensors: ({large_n}, {large_m})")
        large_bin_seq = torch.randn(large_n, large_m, dtype=torch.float32)
        large_mf = torch.randn(large_n, dtype=torch.float32)

        print(f"Input size: {(large_bin_seq.numel() * large_bin_seq.itemsize) / (1024**2):.2f} MB")

        print("\n" + "=" * 80)
        print("All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\nError during example: {e}\n")
