"""
Tensor Concatenation Module
=============================================================================

A high-performance module for reshaping and concatenating CPU tensors.

This module handles the reshaping of 3D and 2D tensors into 2D and 1D tensors, respectively, and
concatenates them along the first dimension with support for both CPU operations.

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Device Placement:
  - All operations are performed on CPU
  - Sign and magnitude handling follows Python/NumPy conventions

Performance Considerations:
  - Memory layout is preserved through careful tensor operations
"""
from logging import getLogger
from typing import Tuple

import torch

# get logger
logger = getLogger(__name__)

# ============================================================================
# Tensor Concatenation Module
# ============================================================================

class TensorConcatenator:
    """High-performance tensor concatenation and reshaping engine.

    This class handles reshaping of 3D and 2D tensors into 2D and 1D tensors,
    respectively, and concatenates them along the first dimension with optional
    GPU acceleration and performance profiling.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Data type for operations. Default is torch.float32.

    Attributes
    ----------
    dtype : torch.dtype
        Data type for operations.

    Examples
    --------
    >>> concatenator = TensorConcatenator()
    >>> src_arr_2d = torch.randn(100, 64)
    >>> tar_arr_3d = torch.randn(10, 8, 64)
    >>> accum_arr_2d = concatenator.concat_arrays(src_arr_2d, tar_arr_3d)
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize tensor concatenator.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Data type for operations. Default is float32.
        """
        logger.info("Initializing TensorConcatenator")

        self.dtype = dtype
        self.device = torch.device("cpu")

        logger.info(f"TensorConcatenator initialized: device={self.device}, dtype={dtype}")

    def _validate_tensor_device(self, tensor: torch.Tensor, name: str) -> None:
        """
        Validate that a tensor is on CPU.

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
                f"Device mismatch for {name}: expected 'cpu', "
                f"got '{tensor.device.type}'"
            )
            raise TypeError(
                f"The {name} does not reside on CPU device. "
                f"Found: {tensor.device.type}"
            )

    def _validate_tensor_shapes(
        self,
        src_arr_2d: torch.Tensor,
        src_mf_1d: torch.Tensor,
        tar_arr_3d: torch.Tensor,
        tar_mf_2d: torch.Tensor
    ) -> None:
        """
        Validate input tensor shapes and consistency.

        Parameters
        ----------
        src_arr_2d : torch.Tensor
            Source 2D binary sequences (N, L).
        src_mf_1d : torch.Tensor
            Source 1D merit factors (N,).
        tar_arr_3d : torch.Tensor
            Target 3D binary sequences (B, M, L).
        tar_mf_2d : torch.Tensor
            Target 2D merit factors (B, M).

        Raises
        ------
        ValueError
            If tensor shapes are incompatible.
        """
        logger.debug(f"Validating tensor shapes:")
        logger.debug(f"  src_arr_2d: {src_arr_2d.shape}")
        logger.debug(f"  src_mf_1d: {src_mf_1d.shape}")
        logger.debug(f"  tar_arr_3d: {tar_arr_3d.shape}")
        logger.debug(f"  tar_mf_2d: {tar_mf_2d.shape}")

        # Check src_arr_2d dimensions
        if src_arr_2d.ndim != 2:
            logger.error(f"src_arr_2d must be 2D, got {src_arr_2d.ndim}D")
            raise ValueError(f"src_arr_2d must be 2D tensor, got shape {src_arr_2d.shape}")

        # Check src_mf_1d dimensions
        if src_mf_1d.ndim != 1:
            logger.error(f"src_mf_1d must be 1D, got {src_mf_1d.ndim}D")
            raise ValueError(f"src_mf_1d must be 1D tensor, got shape {src_mf_1d.shape}")

        # Check tar_arr_3d dimensions
        if tar_arr_3d.ndim != 3:
            logger.error(f"tar_arr_3d must be 3D, got {tar_arr_3d.ndim}D")
            raise ValueError(f"tar_arr_3d must be 3D tensor, got shape {tar_arr_3d.shape}")

        # Check tar_mf_2d dimensions
        if tar_mf_2d.ndim != 2:
            logger.error(f"tar_mf_2d must be 2D, got {tar_mf_2d.ndim}D")
            raise ValueError(f"tar_mf_2d must be 2D tensor, got shape {tar_mf_2d.shape}")

        # Check consistency: src_arr_2d[0] should match tar_arr_3d[-1]
        if src_arr_2d.shape[0] != src_mf_1d.shape[0]:
            logger.error(
                f"src_arr_2d and src_mf_1d first dimension mismatch: "
                f"{src_arr_2d.shape[0]} vs {src_mf_1d.shape[0]}"
            )
            raise ValueError(
                f"src_arr_2d and src_mf_1d must have same first dimension. "
                f"Got {src_arr_2d.shape[0]} vs {src_mf_1d.shape[0]}"
            )

        # Check consistency: tar_arr_3d and tar_mf_2d
        if tar_arr_3d.shape[0] != tar_mf_2d.shape[0] or tar_arr_3d.shape[1] != tar_mf_2d.shape[1]:
            logger.error(
                f"tar_arr_3d and tar_mf_2d shape mismatch: "
                f"{tar_arr_3d.shape[:2]} vs {tar_mf_2d.shape}"
            )
            raise ValueError(
                f"tar_arr_3d and tar_mf_2d first two dimensions must match. "
                f"Got {tar_arr_3d.shape[:2]} vs {tar_mf_2d.shape}"
            )

        # Check consistency: sequence lengths
        if src_arr_2d.shape[1] != tar_arr_3d.shape[2]:
            logger.error(
                f"Sequence length mismatch: src_arr_2d={src_arr_2d.shape[1]}, "
                f"tar_arr_3d={tar_arr_3d.shape[2]}"
            )
            raise ValueError(
                f"Sequence lengths must match. src_arr_2d: {src_arr_2d.shape[1]}, "
                f"tar_arr_3d: {tar_arr_3d.shape[2]}"
            )

        logger.debug("All shape validations passed")

    def concat_arrays(
        self,
        src_arr_2d: torch.Tensor,
        tar_arr_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate source 2D and reshaped target 3D binary sequences.

        This method reshapes the 3D target tensor into 2D and concatenates it
        with the source 2D tensor along the first dimension.

        Parameters
        ----------
        src_arr_2d : torch.Tensor
            Source 2D binary sequences of shape (N, L).
        tar_arr_3d : torch.Tensor
            Target 3D binary sequences of shape (B, M, L).

        Returns
        -------
        torch.Tensor
            Concatenated 2D array of shape (N + B*M, L).

        Raises
        ------
        TypeError
            If tensors are not on CPU.
        ValueError
            If tensor shapes are incompatible.

        Notes
        -----
        The reshaping operation is NOT in-place. torch.cat() creates a new tensor.

        Examples
        --------
        >>> concatenator = TensorConcatenator()
        >>> src = torch.randn(100, 64)
        >>> tar = torch.randn(10, 5, 64)
        >>> result = concatenator.concat_arrays(src, tar)
        >>> assert result.shape == (150, 64)
        """
        logger.info("Starting array concatenation")

        # Validate devices
        self._validate_tensor_device(src_arr_2d, "src_arr_2d")
        self._validate_tensor_device(tar_arr_3d, "tar_arr_3d")

        # Validate shapes
        if src_arr_2d.ndim != 2:
            logger.error(f"src_arr_2d must be 2D, got {src_arr_2d.ndim}D")
            raise ValueError(f"src_arr_2d must be 2D tensor, got shape {src_arr_2d.shape}")

        if tar_arr_3d.ndim != 3:
            logger.error(f"tar_arr_3d must be 3D, got {tar_arr_3d.ndim}D")
            raise ValueError(f"tar_arr_3d must be 3D tensor, got shape {tar_arr_3d.shape}")

        # Check consistency
        if src_arr_2d.shape[1] != tar_arr_3d.shape[2]:
            logger.error(
                f"Sequence length mismatch: src={src_arr_2d.shape[1]}, tar={tar_arr_3d.shape[2]}"
            )
            raise ValueError(
                f"Sequence lengths must match. src_arr_2d: {src_arr_2d.shape[1]}, "
                f"tar_arr_3d: {tar_arr_3d.shape[2]}"
            )

        tar_arr_2d = tar_arr_3d.reshape(-1, tar_arr_3d.shape[2])

        logger.debug(f"Reshaped tar_arr_3d from {tar_arr_3d.shape} to {tar_arr_2d.shape}")

        accum_arr_2d = torch.cat([src_arr_2d, tar_arr_2d], dim=0)

        # Ensure contiguity for downstream operations
        accum_arr_2d = accum_arr_2d.contiguous()

        logger.info(
            f"Output shape: {accum_arr_2d.shape}"
        )

        return accum_arr_2d

    def concat_merit_factors(
        self,
        src_mf_1d: torch.Tensor,
        tar_mf_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate source 1D and reshaped target 2D merit factors.

        This method reshapes the 2D target merit factor tensor into 1D
        and concatenates it with the source 1D tensor along the first dimension.

        Parameters
        ----------
        src_mf_1d : torch.Tensor
            Source 1D merit factors of shape (N,).
        tar_mf_2d : torch.Tensor
            Target 2D merit factors of shape (B, M).

        Returns
        -------
        torch.Tensor
            Concatenated 1D merit factors of shape (N + B*M,).

        Raises
        ------
        TypeError
            If tensors are not on CPU.
        ValueError
            If tensor shapes are incompatible.

        Notes
        -----
        The reshaping operation is NOT in-place. torch.cat() creates a new tensor.

        Examples
        --------
        >>> concatenator = TensorConcatenator()
        >>> src_mf = torch.randn(100)
        >>> tar_mf = torch.randn(10, 5)
        >>> result = concatenator.concat_merit_factors(src_mf, tar_mf)
        >>> assert result.shape == (150,)
        """
        logger.info("Starting merit factor concatenation")

        # Validate devices
        self._validate_tensor_device(src_mf_1d, "src_mf_1d")
        self._validate_tensor_device(tar_mf_2d, "tar_mf_2d")

        # Validate shapes
        if src_mf_1d.ndim != 1:
            logger.error(f"src_mf_1d must be 1D, got {src_mf_1d.ndim}D")
            raise ValueError(f"src_mf_1d must be 1D tensor, got shape {src_mf_1d.shape}")

        if tar_mf_2d.ndim != 2:
            logger.error(f"tar_mf_2d must be 2D, got {tar_mf_2d.ndim}D")
            raise ValueError(f"tar_mf_2d must be 2D tensor, got shape {tar_mf_2d.shape}")

        tar_mf_1d = tar_mf_2d.reshape(-1)

        logger.debug(f"Reshaped tar_mf_2d from {tar_mf_2d.shape} to {tar_mf_1d.shape}")

        accum_mf_1d = torch.cat([src_mf_1d, tar_mf_1d], dim=0)

        # Ensure contiguity for downstream operations
        accum_mf_1d = accum_mf_1d.contiguous()

        return accum_mf_1d

    def concat_tensors(
        self,
        src_arr_2d: torch.Tensor,
        src_mf_1d: torch.Tensor,
        tar_arr_3d: torch.Tensor,
        tar_mf_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape 3D and 2D CPU tensors into 2D and 1D tensors and concatenate them.

        This is the main interface that combines array and merit factor concatenation.
        It performs comprehensive validation and profiling.

        Parameters
        ----------
        src_arr_2d : torch.Tensor
            Source 2D binary sequences of shape (N, L).
        src_mf_1d : torch.Tensor
            Source 1D merit factors of shape (N,).
        tar_arr_3d : torch.Tensor
            Target 3D binary sequences of shape (B, M, L).
        tar_mf_2d : torch.Tensor
            Target 2D merit factors of shape (B, M).

        Returns
        -------
        tuple of torch.Tensor
            - accum_arr_2d: Concatenated 2D binary sequences of shape (N + B*M, L).
            - accum_mf_1d: Concatenated 1D merit factors of shape (N + B*M,).

        Raises
        ------
        TypeError
            If any tensor is not on CPU.
        ValueError
            If tensor shapes are incompatible.

        Notes
        -----
        Sign and magnitude handling:
            - For CPU tensors: All sign and magnitude operations follow Python/NumPy conventions
            - For GPU tensors (if enabled): Bitwise operations handle sign/magnitude automatically
            - No explicit sign/magnitude conversion is needed

        Examples
        --------
        >>> concatenator = TensorConcatenator()
        >>> src_arr = torch.randn(100, 64)
        >>> src_mf = torch.randn(100)
        >>> tar_arr = torch.randn(10, 5, 64)
        >>> tar_mf = torch.randn(10, 5)
        >>> accum_arr, accum_mf = concatenator.concat_tensors(
        ...     src_arr, src_mf, tar_arr, tar_mf
        ... )
        >>> print(f"Arrays: {accum_arr.shape}, Merit Factors: {accum_mf.shape}")
        """
        logger.info("Starting complete tensor concatenation")

        # Validate all inputs
        self._validate_tensor_shapes(src_arr_2d, src_mf_1d, tar_arr_3d, tar_mf_2d)

        # Concatenate both components
        accum_arr_2d = self.concat_arrays(src_arr_2d, tar_arr_3d)
        accum_mf_1d = self.concat_merit_factors(src_mf_1d, tar_mf_2d)

        logger.info(
            f"Output shapes: arrays={accum_arr_2d.shape}, merit_factors={accum_mf_1d.shape}"
        )

        return accum_arr_2d, accum_mf_1d

if __name__ == "__main__":
    """Comprehensive example demonstrating tensor concatenation usage."""

    print("\n" + "=" * 80)
    print("Tensor Concatenation Module - Example Usage")
    print("=" * 80)

    try:
        # Step 1: Create concatenator
        print("\n[Step 2] Initializing Tensor Concatenator")
        print("-" * 80)

        concatenator = TensorConcatenator(
            dtype=torch.float32
        )

        logger.info("Tensor concatenator successfully initialized")

        # Step 2: Create test tensors
        print("\n[Step 3] Creating Test Tensors")
        print("-" * 80)

        # Test case 1: Small tensors
        src_arr_2d_1 = torch.randn(100, 64, dtype=torch.float32)
        src_mf_1d_1 = torch.randn(100, dtype=torch.float32)
        tar_arr_3d_1 = torch.randn(10, 8, 64, dtype=torch.float32)
        tar_mf_2d_1 = torch.randn(10, 8, dtype=torch.float32)

        print(f"\nTest Case 1: Small Tensors")
        print(f"  src_arr_2d: {src_arr_2d_1.shape}")
        print(f"  src_mf_1d:  {src_mf_1d_1.shape}")
        print(f"  tar_arr_3d: {tar_arr_3d_1.shape}")
        print(f"  tar_mf_2d:  {tar_mf_2d_1.shape}")

        # Step 3: Perform concatenation
        print("\n[Step 4] Performing Tensor Concatenation")
        print("-" * 80)

        accum_arr_2d_1, accum_mf_1d_1 = concatenator.concat_tensors(
            src_arr_2d_1, src_mf_1d_1, tar_arr_3d_1, tar_mf_2d_1
        )

        print(f"\nResults:")
        print(f"  accum_arr_2d: {accum_arr_2d_1.shape}")
        print(f"  accum_mf_1d:  {accum_mf_1d_1.shape}")

        # Test case 2: Larger tensors
        print("\n[Step 6] Testing with Larger Tensors")
        print("-" * 80)

        src_arr_2d_2 = torch.randn(1000, 256, dtype=torch.float32)
        src_mf_1d_2 = torch.randn(1000, dtype=torch.float32)
        tar_arr_3d_2 = torch.randn(50, 20, 256, dtype=torch.float32)
        tar_mf_2d_2 = torch.randn(50, 20, dtype=torch.float32)

        print(f"\nTest Case 2: Large Tensors")
        print(f"  src_arr_2d: {src_arr_2d_2.shape}")
        print(f"  src_mf_1d:  {src_mf_1d_2.shape}")
        print(f"  tar_arr_3d: {tar_arr_3d_2.shape}")
        print(f"  tar_mf_2d:  {tar_mf_2d_2.shape}")

        accum_arr_2d_2, accum_mf_1d_2 = concatenator.concat_tensors(
            src_arr_2d_2, src_mf_1d_2, tar_arr_3d_2, tar_mf_2d_2
        )

        print(f"\nResults:")
        print(f"  accum_arr_2d: {accum_arr_2d_2.shape}")
        print(f"  accum_mf_1d:  {accum_mf_1d_2.shape}")

        # Step 5: Verify correctness
        print("\n[Step 7] Verification")
        print("-" * 80)

        expected_arr_rows = src_arr_2d_1.shape[0] + tar_arr_3d_1.shape[0] * tar_arr_3d_1.shape[1]
        expected_mf_size = src_mf_1d_1.shape[0] + tar_mf_2d_1.shape[0] * tar_mf_2d_1.shape[1]

        assert accum_arr_2d_1.shape[0] == expected_arr_rows, "Array row count mismatch!"
        assert accum_mf_1d_1.shape[0] == expected_mf_size, "Merit factor size mismatch!"
        assert accum_arr_2d_1.is_contiguous(), "Output array not contiguous!"
        assert accum_mf_1d_1.is_contiguous(), "Output merit factors not contiguous!"

        print(f" All assertions passed!")
        print(f" Output tensors are contiguous")
        print(f" Shapes are correct")

        print("\n" + "=" * 80)
        print("Example Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\nError during example: {e}\n")
