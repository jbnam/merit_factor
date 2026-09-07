"""
Bernoulli Distribution Module
=====================================

A high-performance module for computing Bernoulli distributions from binary
sequences using bitwise operations with comprehensive logging and validation
and generating 2D tensors of random limbs with bitwise Bernoulli distributions.

This module calculates success rates (p_k) for Bernoulli distributions based on
the bit patterns in elite binary sequences, enabling adaptive probability updates
for evolutionary algorithms and provides GPU-accelerated generation of random
binary sequences with Bernoulli-distributed bits, optimized for NVIDIA GPUs.

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Bitwise Operations:
  - All bitwise operations are performed on CPU in this module
  - Sign and magnitude handling follows standard integer semantics
  - Each element in the output array is success rate (probability), always in range [0.0, 1.0]

Performance:
  - Uses PyTorch native operations for maximum efficiency
  - Memory layout optimization through contiguity enforcement
  - Vectorized bitwise operations avoid loops

Example:
    >>> generator = BernoulliLimbGenerator(device="cuda")
    >>> arr_p = torch.tensor([0.5] * 96, dtype=torch.float32, device="cuda")
    >>> result = generator.generate(num_arrays=10, num_limbs=3, arr_p=arr_p)
    >>> print(result.shape)
    torch.Size([10, 3])
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

import src.common as cm

# Initialize logger
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BernoulliDistributionConfig:
    """Configuration parameters for Bernoulli distribution computation.

    Attributes
    ----------
    len_bin_seq : int
      The bit length of a binary sequence (total bits).
    validate_input : bool
      Whether to perform comprehensive input validation. Default is True.
    ensure_contiguous : bool
      Whether to ensure tensors are contiguous in memory. Default is True.
    """
    len_bin_seq: int
    validate_input: bool = True
    ensure_contiguous: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.len_bin_seq < cm.MIN_BIT_LEN or self.len_bin_seq > cm.MAX_BIT_LEN:
            raise ValueError(f"len_bin_seq must be between {cm.MIN_BIT_LEN} and {cm.MAX_BIT_LEN}, got {self.len_bin_seq}")

# ============================================================================
# Bernoulli Distribution Computer
# ============================================================================

class BernoulliDistributionComputer:
    """High-performance Bernoulli distribution computation engine.

    This class computes success rates (p_k) for Bernoulli distributions from
    binary sequences using optimized bitwise operations. Each position k in
    the output distribution represents the proportion of sequences with bit 1
    at that position.

    Parameters
    ----------
    config : BernoulliDistributionConfig, optional
        Configuration parameters. If None, creates minimal config.

    Attributes
    ----------
    config : BernoulliDistributionConfig
        Configuration parameters.

    Examples
    --------
    >>> config = BernoulliDistributionConfig(len_bin_seq=256)
    >>> computer = BernoulliDistributionComputer(config=config)
    >>> bin_seq_2d = torch.randn(1000, 8) > 0.5  # 8 limbs of 32 bits each
    >>> distribution = computer.compute_distribution(bin_seq_2d)
    >>> print(f"Distribution shape: {distribution.shape}")
    """

    def __init__(self, config: Optional[BernoulliDistributionConfig] = None):
        """Initialize Bernoulli distribution computer.

        Parameters
        ----------
        config : BernoulliDistributionConfig, optional
            Configuration parameters. Creates default if None.
        """
        if config is None:
            logger.warning("No config provided, using default configuration")
            config = BernoulliDistributionConfig(len_bin_seq=cm.MIN_BIT_LEN)

        self.config = config

        logger.info(
            f"Initialized BernoulliDistributionComputer: "
            f"len_bin_seq={config.len_bin_seq}"
        )

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
                f"The {name} must reside on CPU. "
                f"Found: {tensor.device.type}"
            )

    def _validate_dimensions(self, bin_seq_2d: torch.Tensor) -> None:
        """
        Validate tensor dimensions.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            Binary sequences tensor.

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

    def _validate_shape_consistency(self, bin_seq_2d: torch.Tensor) -> None:
        """
        Validate shape consistency with configuration.

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            Binary sequences tensor of shape (N, M) where M is number of limbs.

        Raises
        ------
        ValueError
            If shape is inconsistent with len_bin_seq.
        """
        n_sequences, n_limbs = bin_seq_2d.shape

        min_bits = cm.LIMB_BIT_SIZE * (n_limbs - 1)
        max_bits = cm.LIMB_BIT_SIZE * n_limbs

        if not (min_bits < self.config.len_bin_seq <= max_bits):
            logger.error(
                f"Bit length constraint violation: len_bin_seq={self.config.len_bin_seq} "
                f"not in range ({min_bits}, {max_bits}] for {n_limbs} limbs"
            )
            raise ValueError(
                f"Inconsistent bit length constraint {self.config.len_bin_seq} "
                f"for {n_limbs} limbs with {cm.LIMB_BIT_SIZE} bits/limb. "
                f"Valid range: ({min_bits}, {max_bits}]"
            )

        logger.debug(
            f"Shape validation passed: n_sequences={n_sequences}, "
            f"n_limbs={n_limbs}, len_bin_seq={self.config.len_bin_seq}"
        )

    def _create_bit_masks(self) -> torch.Tensor:
        """
        Create bit masks for extracting individual bits from limbs.

        Returns
        -------
        torch.Tensor
            1D tensor of shape (cm.LIMB_BIT_SIZE,) containing bit masks
            in descending order (MSB to LSB).

        Notes
        -----
        For cm.LIMB_BIT_SIZE=32, creates masks:
        [2^31, 2^30, ..., 2^1, 2^0]
        """
        logger.debug(f"Creating bit masks for {cm.LIMB_BIT_SIZE} bits")

        bit_positions = range(cm.LIMB_BIT_SIZE - 1, -1, -1)
        mask_values = [1 << j for j in bit_positions]

        # Create tensor on CPU
        masks = torch.tensor(
            mask_values,
            dtype=cm.T_INT64,  # Use int64 to safely represent large powers of 2
            device="cpu"
        )

        logger.debug(f"Created bit mask tensor: shape={masks.shape}, device={masks.device}")

        return masks

    def compute_distribution(
        self,
        bin_seq_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Bernoulli distribution (success rates) from binary sequences.

        This method calculates the proportion of 1-bits at each position across
        all sequences. For each bit position k, the output is the count of
        sequences with bit 1 at position k, divided by the total number of
        sequences.

        Algorithm:
        1. Create bit masks for extracting individual bits
        2. Apply bitwise AND to extract each bit position
        3. Count set bits (population count) using numpy
        4. Sum counts across sequences
        5. Normalize by number of sequences to get probabilities

        Parameters
        ----------
        bin_seq_2d : torch.Tensor
            2D tensor of binary sequences with shape (N, M) where:
            - N is the number of sequences
            - M is the number of limbs (typically 32-bit integers)

        Returns
        -------
        torch.Tensor
            1D tensor of shape (len_bin_seq,) containing success rates (probabilities)
            for each bit position, in range [0.0, 1.0].

        Raises
        ------
        TypeError
            If bin_seq_2d is not on CPU or has unsupported dtype.
        ValueError
            If bin_seq_2d has invalid dimensions or shape inconsistencies.

        Notes
        -----
        Sign and Magnitude:
            - Input tensors use standard integer bit representation
            - Bitwise operations preserve bit patterns
            - Output is always non-negative (probability)

        Performance Notes:
            - Time complexity: O(N * M * B) where B = cm.LIMB_BIT_SIZE
            - Memory complexity: O(N * M * B) for intermediate results
            - Uses vectorized operations, no explicit loops

        Examples
        --------
        >>> config = BernoulliDistributionConfig(len_bin_seq=256)
        >>> computer = BernoulliDistributionComputer(config=config)
        >>> bin_seq_2d = torch.randint(0, 2**32, (1000, 8), dtype=torch.int64)
        >>> distribution = computer.compute_distribution(bin_seq_2d)
        >>> print(f"Distribution: min={distribution.min():.3f}, max={distribution.max():.3f}")
        >>> assert distribution.shape == (256,)
        >>> assert (distribution >= 0).all() and (distribution <= 1).all()
        """
        logger.info(f"Starting distribution computation: input shape={bin_seq_2d.shape}")

        total_start = time.time()

        # Input validation
        if self.config.validate_input:
            logger.debug("Performing input validation")

            self._validate_tensor_device(bin_seq_2d, "bin_seq_2d")
            self._validate_dimensions(bin_seq_2d)
            self._validate_shape_consistency(bin_seq_2d)

            logger.debug("Input validation completed")

        # Ensure contiguous memory layout
        if self.config.ensure_contiguous and not bin_seq_2d.is_contiguous():
            logger.debug("Making bin_seq_2d contiguous")
            bin_seq_2d = bin_seq_2d.contiguous()

        # Get dimensions
        n_sequences, n_limbs = bin_seq_2d.shape

        logger.info(
            f"Processing: n_sequences={n_sequences}, n_limbs={n_limbs}, "
            f"len_bin_seq={self.config.len_bin_seq}"
        )

        # Convert to int64 if necessary for bitwise operations
        if bin_seq_2d.dtype != cm.T_INT64:
            logger.debug(f"Converting {bin_seq_2d.dtype} to int64 for bitwise operations")
            bin_seq_2d = bin_seq_2d.to(cm.T_INT64)

        # Step 1: Create bit masks
        logger.debug("Step 1: Creating bit masks")
        masks = self._create_bit_masks()  # Shape: (cm.LIMB_BIT_SIZE,)

        # Step 2: Apply bitwise AND to extract each bit
        logger.debug("Step 2: Performing bitwise AND operations")
        bitwise_start = time.time()

        # Reshape for broadcasting: bin_seq_2d (N, M, 1) × masks (1, 1, B)
        bin_seq_expanded = bin_seq_2d.unsqueeze(2)  # Shape: (N, M, 1)
        masks_expanded = masks.view(1, 1, -1)  # Shape: (1, 1, B)

        masked_bits = torch.bitwise_and(bin_seq_expanded, masks_expanded)  # Shape: (N, M, B)

        bitwise_time = time.time() - bitwise_start
        logger.debug(f"Bitwise AND completed in {bitwise_time:.6f}s, shape={masked_bits.shape}")

        # Step 3: Convert to numpy and use numpy's bitwise_count for efficiency
        logger.debug("Step 3: Computing population counts")
        popcount_start = time.time()

        masked_np = masked_bits.numpy()

        try:
            bit_counts = np.bitwise_count(masked_np.astype(np.uint64))
        except AttributeError:
            # Fallback for older NumPy versions
            logger.warning("np.bitwise_count not available, using fallback")
            bit_counts = np.array([cm.N_INT64(masked_np == masks[i].item()).sum(axis=(0, 1))
                                   for i in range(len(masks))])

        # Convert back to tensor
        bit_counts_tensor = torch.from_numpy(bit_counts).to(cm.T_FLOAT32)

        popcount_time = time.time() - popcount_start
        logger.debug(f"Population count completed in {popcount_time:.6f}s")

        # Step 4: Sum across sequences (along dimension 0)
        logger.debug("Step 4: Reducing across sequences")
        sum_start = time.time()

        bit_sum_per_position = torch.sum(bit_counts_tensor, dim=0)  # Shape: (B,)

        sum_time = time.time() - sum_start
        logger.debug(f"Reduction completed in {sum_time:.6f}s, shape={bit_sum_per_position.shape}")

        # Step 5: Flatten and normalize
        logger.debug("Step 5: Flattening and normalizing")

        flattened_sum = torch.flatten(bit_sum_per_position)

        truncated_sum = flattened_sum[:self.config.len_bin_seq]

        # Normalize by number of sequences to get probabilities
        if n_sequences == 0:
            logger.error("Zero sequences in input")
            raise ValueError("Cannot compute distribution from zero sequences")

        distribution = truncated_sum / n_sequences
        distribution = distribution.to(cm.T_FLOAT32)

        # Sanity check of the output range of distribution
        if not torch.all((distribution >= 0.0) & (distribution <= 1.0)):
            logger.error("Distribution values out of range [0.0, 1.0]")
            raise ValueError("Computed distribution has values outside [0.0, 1.0]")

        total_time = time.time() - total_start

        return distribution

# ============================================================================
# Legacy API (Backward Compatibility)
# ============================================================================

def compute_distributions(
    bin_seq_2d: torch.Tensor,
    len_bin_seq: int,
) -> torch.Tensor:
    """
    Calculate Bernoulli distributions from binary sequences
    (mostly the elites for merit factor).

    This is the legacy function interface maintained for backward compatibility.
    It wraps the BernoulliDistributionComputer class.

    Parameters
    ----------
    bin_seq_2d : torch.Tensor
        2D tensor of binary sequences with shape (N, M) containing elite
        binary sequences. Should be on CPU and have integer dtype.
    len_bin_seq : int
        The total bit length of a binary sequence (e.g., 256 bits).

    Returns
    -------
    torch.Tensor
        1D tensor of shape (len_bin_seq,) containing success rates (probabilities)
        for each bit position, representing the Bernoulli distribution parameters.

    Raises
    ------
    TypeError
        If bin_seq_2d is not on CPU.
    ValueError
        If bin_seq_2d dimensions are invalid or shape is inconsistent.

    Notes
    -----
    Output Values:
        - Each element p_k ∈ [0.0, 1.0] represents the probability of bit 1
          at position k across all sequences
        - p_k = (count of sequences with bit 1 at position k) / (total sequences)

    Examples
    --------
    >>> bin_seq_2d = torch.randint(0, 2**32, (1000, 8), dtype=torch.int64)
    >>> distribution = compute_distributions(bin_seq_2d, len_bin_seq=256)
    >>> print(f"Distribution shape: {distribution.shape}")
    >>> print(f"Min probability: {distribution.min():.3f}")
    >>> print(f"Max probability: {distribution.max():.3f}")
    """
    logger.info(
        f"Legacy compute_distributions called: "
        f"input_shape={bin_seq_2d.shape}, len_bin_seq={len_bin_seq}, "
    )

    # Create configuration and computer
    config = BernoulliDistributionConfig(
        len_bin_seq=len_bin_seq,
        validate_input=True
    )

    computer = BernoulliDistributionComputer(config=config)
    return computer.compute_distribution(bin_seq_2d)


# ============================================================================
# BERNOULLI LIMB GENERATOR
# ============================================================================

class BernoulliLimbGenerator:
    """
    High-performance GPU-based generator for random integer limbs with bitwise Bernoulli distributions.

    This class generates 2D tensors of random 32-bit signed integers where each bit
    independently follows a Bernoulli distribution. The sign bit (MSB) of the first
    limb is fixed to 0 for merit factor symmetry.

    **Important**: Internally, operations treat int32 as uint32 for bitwise operations.
    Sign extension may occur during CPU transfer; this is mitigated through proper
    masking operations on GPU.

    Attributes:
        device (str): CUDA device identifier.
        gpu_info (GPUArchitectureInfo): GPU architecture information.

    Example:
        >>> generator = BernoulliLimbGenerator(device="cuda")
        >>> arr_p = torch.tensor([0.5] * 96, dtype=torch.float32, device="cuda")
        >>> result = generator.generate(num_arrays=10, num_limbs=3, arr_p=arr_p)
        >>> print(result.shape, result.dtype)
        torch.Size([10, 3]) torch.int32
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the Bernoulli limb generator.

        Args:
            device (str): CUDA device identifier (default: "cuda").

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This generator requires GPU support.")

        self.device = device

    def _validate_inputs(self, num_arrays: int, num_limbs: int,
                        arr_p: torch.Tensor) -> None:
        """
        Validate input parameters.

        Args:
            num_arrays (int): Number of random arrays to generate.
            num_limbs (int): Number of 32-bit limbs per array.
            arr_p (torch.Tensor): 1D tensor of Bernoulli success probabilities.

        Raises:
            ValueError: If inputs are invalid.
        """
        len_bin_seq = len(arr_p)
        total_bits = cm.LIMB_BIT_SIZE * num_limbs

        if len_bin_seq <= total_bits - cm.LIMB_BIT_SIZE:
            logger.warning(
                f"Array length {len_bin_seq} is insufficient for {num_limbs} limbs. "
                f"Minimum required: {total_bits - cm.LIMB_BIT_SIZE + 1}"
            )

        if len_bin_seq > total_bits:
            raise ValueError(
                f"Bernoulli array length {len_bin_seq} exceeds available bits {total_bits}"
            )

        if num_arrays <= 0 or num_limbs <= 0:
            raise ValueError(f"num_arrays ({num_arrays}) and num_limbs ({num_limbs}) must be positive")

        if arr_p.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"arr_p must have dtype float32 or float64, got {arr_p.dtype}")

        # Check probability bounds to avoid undefined behavior in Bernoulli
        if torch.any((arr_p < 0.0) | (arr_p > 1.0)):
            logger.error("Probabilities in arr_p must be in [0.0, 1.0]")
            raise ValueError("All probabilities must be in [0.0, 1.0]")

        logger.debug(
            f"Validation passed: num_arrays={num_arrays}, num_limbs={num_limbs}, "
            f"arr_p_len={len_bin_seq}, total_bits={total_bits}"
        )

    def generate(self, num_arrays: int, num_limbs: int,
                arr_p: torch.Tensor) -> torch.Tensor:
        """
        Generate random limbs with bitwise Bernoulli distributions.

        The leading bit (MSB) of the first limb in each array is fixed to 0
        for merit factor symmetry.

        Args:
            num_arrays (int): Number of random arrays to generate.
            num_limbs (int): Number of cm.LIMB_BIT_SIZE-bit limbs per array.
            arr_p (torch.Tensor): 1D tensor of Bernoulli success rates for each bit position.
                                 Shape: (num_bits,) where num_bits <= cm.LIMB_BIT_SIZE * num_limbs.

        Returns:
            torch.Tensor: 2D tensor of shape (num_arrays, num_limbs) with dtype=int32.
                         Each element represents a cm.LIMB_BIT_SIZE-bit signed integer (treated as
                         cm.LIMB_BIT_SIZE-bit unsigned for bitwise operations).

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If GPU operations fail.

        Example:
            >>> gen = BernoulliLimbGenerator("cuda")
            >>> probs = torch.tensor([0.5] * 96, dtype=torch.float32, device="cuda")
            >>> limbs = gen.generate(num_arrays=5, num_limbs=3, arr_p=probs)
            >>> print(limbs.shape)
            torch.Size([5, 3])
        """
        logger.info(f"Generating {num_arrays} arrays with {num_limbs} limbs each")

        self._validate_inputs(num_arrays, num_limbs, arr_p)

        len_bin_seq = len(arr_p)
        total_bits = cm.LIMB_BIT_SIZE * num_limbs
        pad_size = total_bits - len_bin_seq

        try:
            # Ensure input is on correct device and dtype
            arr_p = arr_p.to(device=self.device, dtype=cm.T_FLOAT32)

            logger.debug(f"Input preparation: pad_size={pad_size}, total_bits={total_bits}")

            # Generate Bernoulli random bits
            arr_p_expanded = arr_p.unsqueeze(0).expand(num_arrays, len_bin_seq)
            rand_bits = torch.bernoulli(arr_p_expanded).to(dtype=cm.T_INT32)

            logger.debug(f"Generated Bernoulli random bits: shape={rand_bits.shape}")

            # Pad with zeros for the MSB of first limb
            padded_rand_bits = F.pad(
                rand_bits,
                (0, pad_size, 0, 0),
                mode="constant",
                value=0
            )

            logger.debug(f"Padded random bits: shape={padded_rand_bits.shape}")

            # Reshape to separate limbs
            bit_mat = padded_rand_bits.view(num_arrays, num_limbs, cm.LIMB_BIT_SIZE)

            shifter = torch.arange(
                cm.LIMB_BIT_SIZE - 1, -1, -1,
                dtype=cm.T_INT32,
                device=self.device
            )

            logger.debug(f"Bit shifter created: {shifter[:5]}...{shifter[-5:]}")

            # Convert bits to integers via left-shift and summation
            int_batch = (bit_mat << shifter).sum(dim=2, dtype=cm.T_INT32)

            # Force MSB of first limb to 0 for symmetry
            int_batch[:, 0].bitwise_and_(cm.MSB_ZERO_MASK32)

            logger.info(
                f"Successfully generated limbs: shape={int_batch.shape}, "
                f"dtype={int_batch.dtype}, device={int_batch.device}"
            )

            return int_batch

        except RuntimeError as e:
            logger.error(f"GPU operation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate random limbs: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    """Comprehensive example demonstrating Bernoulli distribution computation."""

    print("\n" + "=" * 80)
    print("Bernoulli Distribution Computer - Example Usage")
    print("=" * 80)

    try:
        # Example 1: Basic usage with legacy function
        print("\n[Example 1] Basic Usage with Legacy Function")
        print("-" * 80)

        # Create sample binary sequences
        n_sequences = 1000
        sequence_length = 256
        n_limbs = (sequence_length + cm.LIMB_BIT_SIZE - 1) // cm.LIMB_BIT_SIZE

        # Create random binary sequences
        bin_seq_2d = torch.randint(0, 2**32, (n_sequences, n_limbs), dtype=cm.T_INT64)

        print(f"Input tensors:")
        print(f"  bin_seq_2d shape: {bin_seq_2d.shape}")
        print(f"  bin_seq_2d dtype: {bin_seq_2d.dtype}")
        print(f"  Sequence length: {sequence_length} bits")
        print(f"  Number of limbs: {n_limbs}")

        # Compute distribution using legacy function
        distribution = compute_distributions(
            bin_seq_2d,
            len_bin_seq=sequence_length,
        )

        print(f"\nOutput distribution:")
        print(f"  Shape: {distribution.shape}")
        print(f"  Dtype: {distribution.dtype}")
        print(f"  Min probability: {distribution.min():.6f}")
        print(f"  Max probability: {distribution.max():.6f}")
        print(f"  Mean probability: {distribution.mean():.6f}")
        print(f"  Valid range [0,1]: {(distribution.min() >= 0 and distribution.max() <= 1).item()}")

        # Example 2: Probability range validation
        print("\n[Example 2] Probability Range Validation")
        print("-" * 80)

        # Create highly skewed sequences (mostly zeros)
        skewed_seq = torch.zeros((1000, 8), dtype=cm.T_INT64)
        skewed_seq[0, 0] = 2**32 - 1  # One sequence with all 1s

        dist_skewed = compute_distributions(
            skewed_seq,
            len_bin_seq=256,
        )

        print(f"Skewed distribution (mostly zeros):")
        print(f"  Min: {dist_skewed.min():.6f}")
        print(f"  Max: {dist_skewed.max():.6f}")
        print(f"  Mean: {dist_skewed.mean():.6f}")
        print(f"  Most positions have p≈0.001 (1 sequence out of 1000)")

        # Example 3: Large-scale performance test
        print("\n[Example 3] Large-Scale Performance Test")
        print("-" * 80)

        large_n = 100000
        large_m = 16  # 16 × 32 = 512 bits

        print(f"Creating large tensor: ({large_n}, {large_m})")
        large_seq = torch.randint(0, 2**32, (large_n, large_m), dtype=cm.T_INT64)

        print(f"Input size: {(large_seq.numel() * large_seq.itemsize) / (1024**2):.2f} MB")

        computer_large = BernoulliDistributionComputer(
            BernoulliDistributionConfig(len_bin_seq=512)
        )

        dist_large = computer_large.compute_distribution(large_seq)

        # Example 4: Error handling
        print("\n[Example 4] Error Handling")
        print("-" * 80)

        test_cases = [
            {
                "name": "Invalid dimensions (3D tensor)",
                "tensor": torch.randn(10, 8, 4),
                "should_fail": True
            },
            {
                "name": "Invalid dimensions (1D tensor)",
                "tensor": torch.randn(10),
                "should_fail": True
            },
            {
                "name": "Inconsistent shape (too many limbs)",
                "tensor": torch.randint(0, 2**32, (100, 20), dtype=cm.T_INT64),
                "config": BernoulliDistributionConfig(len_bin_seq=1000),
                "should_fail": True
            },
            {
                "name": "Valid configuration",
                "tensor": torch.randint(0, 2**32, (100, 8), dtype=cm.T_INT64),
                "config": BernoulliDistributionConfig(len_bin_seq=256),
                "should_fail": False
            },
        ]

        for test_case in test_cases:
            try:
                config = test_case.get("config", BernoulliDistributionConfig(len_bin_seq=256))
                computer_test = BernoulliDistributionComputer(config=config)
                dist = computer_test.compute_distribution(test_case["tensor"])

                if test_case["should_fail"]:
                    print(f"   {test_case['name']}: Expected failure but succeeded")
                else:
                    print(f"   {test_case['name']}: Succeeded")
            except (ValueError, TypeError) as e:
                if test_case["should_fail"]:
                    print(f"   {test_case['name']}: Failed as expected")
                else:
                    print(f"   {test_case['name']}: Unexpected failure")

        print("\n" + "=" * 80)
        print("All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\nError during example: {e}\n")
