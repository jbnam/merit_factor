"""
GPU Kernel Dimension Calculator
================================

A high-performance module for computing optimal CUDA kernel launch parameters
based on GPU architecture constraints and workload requirements.

This module dynamically queries GPU attributes and calculates optimal block sizes,
grid sizes, and iteration counts for CUDA kernel launches, taking into account
register pressure and SM occupancy.

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
It assumes that a single GPU is used for computation  (e.g., GTX 1080 Ti).
"""

import logging
import math
from typing import Optional, NamedTuple

import torch

import src.common as cm

# Configure module logger
logger = logging.getLogger(__name__)


class KernelDimensions(NamedTuple):
    """Container for kernel launch dimensions.

    Attributes
    ----------
    grid_size : int
        Number of thread blocks in the grid.
    block_size : int
        Number of threads per block.
    num_iterations : int
        Number of kernel launch iterations required to process all samples.
    """
    grid_size: int
    block_size: int
    num_iterations: int


class GPUArchitectureParams:
    """Encapsulates GPU architecture parameters retrieved dynamically.

    Parameters
    ----------
    device_id : int, optional
        CUDA device ID. Defaults to current device.

    Attributes
    ----------
    num_sm : int
        Number of streaming multiprocessors.
    max_threads_per_block : int
        Maximum threads per block supported by the architecture.
    max_registers_per_sm : int
        Total register file size per SM.
    max_registers_per_thread : int
        Maximum registers allocatable per thread.
    device_name : str
        Name of the GPU device.

    Raises
    ------
    RuntimeError
        If CUDA is not available or device query fails.

    Examples
    --------
    >>> gpu_params = GPUArchitectureParams()
    >>> print(f"GPU: {gpu_params.device_name}, SMs: {gpu_params.num_sm}")
    """
    def __init__(self, device_id: Optional[int] = None):
        """Initialize GPU architecture parameters."""
        if not torch.cuda.is_available():
            error_msg = (
                "CUDA is not available. This module requires a CUDA-capable GPU.\n"
                f"PyTorch version: {torch.__version__}\n"
                f"CUDA built version: {torch.version.cuda if torch.version.cuda else 'CPU-only'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.device_id = device_id if device_id is not None else torch.cuda.current_device()

        try:
            # Query GPU properties
            props = torch.cuda.get_device_properties(self.device_id)

            # ===== ROBUST ATTRIBUTE ACCESS =====
            # Different PyTorch versions use different naming conventions

            # Device name (usually consistent)
            self.device_name = self._get_attr(props, ['name'], 'Unknown GPU')

            # Number of SMs (multiple possible names)
            self.num_sm = self._get_attr(
                props,
                ['multi_processor_count', 'multiProcessorCount'],
                28  # Default for GTX 1080 Ti
            )

            # Max threads per block (multiple possible names)
            self.max_threads_per_block = self._get_attr(
                props,
                ['max_threads_per_block', 'maxThreadsPerBlock'],
                1024  # Safe default
            )

            # Compute capability for determining register limits
            major = self._get_attr(props, ['major'], 6)
            minor = self._get_attr(props, ['minor'], 1)
            self.compute_capability = f"{major}.{minor}"

            # Total memory (for informational purposes)
            self.total_memory = self._get_attr(
                props,
                ['total_memory', 'totalGlobalMem'],
                0
            )

            # ===== REGISTER INFORMATION =====

            # Try to get registers per SM from device
            self.max_registers_per_sm = self._get_attr(
                props,
                ['regs_per_multiprocessor', 'regsPerMultiprocessor'],
                None
            )

            if self.max_registers_per_sm is None:
                # Fallback: determine from compute capability
                self.max_registers_per_sm = self._get_registers_per_sm_from_compute_capability(major, minor)
                logger.info(f"Using architecture default for registers per SM: {self.max_registers_per_sm}")
            else:
                logger.info(f"Retrieved registers per SM from device: {self.max_registers_per_sm}")

            # Get max registers per thread from compute capability
            self.max_registers_per_thread = self._get_max_registers_per_thread(major, minor)

            logger.info(f"Initialized GPU parameters for device {self.device_id}: {self.device_name}")
            logger.debug(f"  Compute Capability: {self.compute_capability}")
            logger.debug(f"  SMs: {self.num_sm}")
            logger.debug(f"  Max threads/block: {self.max_threads_per_block}")
            logger.debug(f"  Max registers/SM: {self.max_registers_per_sm}")
            logger.debug(f"  Max registers/thread: {self.max_registers_per_thread}")
            logger.debug(f"  Warp size: {cm.WARP_SIZE}")

        except Exception as e:
            logger.error(f"Failed to query GPU device {self.device_id}: {e}")
            raise RuntimeError(f"GPU device query failed for device {self.device_id}") from e

    @staticmethod
    def _get_attr(props, attr_names: list, default):
        """
        Try multiple attribute names and return the first one that exists.

        Parameters
        ----------
        props : torch._C._CudaDeviceProperties
            CUDA device properties object.
        attr_names : list of str
            List of possible attribute names to try.
        default : any
            Default value if none of the attributes exist.

        Returns
        -------
        any
            The attribute value or default.
        """
        for attr_name in attr_names:
            if hasattr(props, attr_name):
                value = getattr(props, attr_name)
                logger.debug(f"Found attribute '{attr_name}' with value: {value}")
                return value

        logger.warning(f"None of {attr_names} found. Using default: {default}")
        return default

    @staticmethod
    def _get_registers_per_sm_from_compute_capability(major: int, minor: int) -> int:
        """
        Get registers per SM based on compute capability.

        Source: NVIDIA CUDA Programming Guide
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        Parameters
        ----------
        major : int
            Compute capability major version.
        minor : int
            Compute capability minor version.

        Returns
        -------
        int
            Number of 32-bit registers per SM.
        """
        # Most modern architectures have 65536 registers per SM
        if major >= 5:  # Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper
            return 65536
        elif major == 3:  # Kepler
            return 65536
        elif major == 2:  # Fermi
            return 32768
        else:
            return 65536  # Safe default

    @staticmethod
    def _get_max_registers_per_thread(major: int, minor: int) -> int:
        """
        Get max registers per thread based on compute capability.

        Source: NVIDIA CUDA Programming Guide
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        Parameters
        ----------
        major : int
            Compute capability major version.
        minor : int
            Compute capability minor version.

        Returns
        -------
        int
            Maximum number of 32-bit registers per thread.
        """
        if major >= 5:  # Maxwell and newer (includes Pascal/GTX 1080 Ti)
            return 255
        elif major == 3:  # Kepler
            return 255
        elif major == 2:  # Fermi
            return 63
        else:
            return 255  # Safe default for modern GPUs

    def __repr__(self) -> str:
        """String representation of GPU parameters."""
        return (
            f"GPUArchitectureParams(device={self.device_id}, name='{self.device_name}', "
            f"CC={self.compute_capability}, num_sm={self.num_sm}, "
            f"max_threads_per_block={self.max_threads_per_block}, "
            f"max_regs_per_sm={self.max_registers_per_sm})"
        )


class KernelDimensionCalculator:
    """Calculates optimal CUDA kernel launch dimensions with register-aware optimization.

    This class computes grid sizes, block sizes, and iteration counts based on:
    - GPU architecture constraints (SMs, registers, thread limits)
    - Workload size (number of samples)
    - Register pressure (binary sequence length)

    Parameters
    ----------
    gpu_params : GPUArchitectureParams, optional
        GPU architecture parameters. If None, creates default for current device.

    Examples
    --------
    >>> calculator = KernelDimensionCalculator()
    >>> dims = calculator.compute_dimensions(sample_size=100000, num_limbs=64)
    >>> print(f"Grid: {dims.grid_size}, Block: {dims.block_size}, Iters: {dims.num_iterations}")
    """

    def __init__(self, gpu_params: Optional[GPUArchitectureParams] = None):
        """Initialize the kernel dimension calculator.

        Parameters
        ----------
        gpu_params : GPUArchitectureParams, optional
            Pre-configured GPU parameters. If None, auto-detects current device.
        """
        self.gpu_params = gpu_params if gpu_params is not None else GPUArchitectureParams()
        logger.info(f"KernelDimensionCalculator initialized for {self.gpu_params.device_name}")

    def compute_max_block_size(self, num_limbs: int) -> int:
        """Compute maximum block size constrained by register availability.

        The block size is limited by the total register file size per SM and the
        number of registers required per thread. The result is rounded down to
        the nearest multiple of warp size for optimal performance.

        Parameters
        ----------
        num_limbs : int
            Number of limbs (elements) in the binary sequence per thread.
            Each limb typically represents one register.

        Returns
        -------
        int
            Maximum block size (threads per block) achievable with given register usage.
            Guaranteed to be a multiple of warp size.

        Raises
        ------
        ValueError
            If num_limbs is non-positive or exceeds architectural limits.

        Notes
        -----
        The formula used is:

        .. math::

            \\text{max\\_block\\_size} = \\left\\lfloor \\frac{\\text{registers\\_per\\_SM}}{\\text{registers\\_per\\_thread}} \\right\\rfloor

        where registers_per_thread = num_limbs + cm.NUM_REG_PER_THREAD.

        Examples
        --------
        >>> calculator = KernelDimensionCalculator()
        >>> max_bs = calculator.compute_max_block_size(num_limbs=64)
        >>> print(f"Max block size with 64 limbs: {max_bs}")
        """
        # Input validation
        if num_limbs <= 0:
            logger.error(f"Invalid num_limbs: {num_limbs}. Must be positive.")
            raise ValueError(f"num_limbs must be positive, got {num_limbs}")

        max_limbs_allowed = (self.gpu_params.max_registers_per_thread -
                             cm.NUM_REG_PER_THREAD)

        if num_limbs > max_limbs_allowed:
            logger.error(f"num_limbs {num_limbs} exceeds maximum allowed {max_limbs_allowed}")
            raise ValueError(
                f"num_limbs ({num_limbs}) exceeds maximum allowed ({max_limbs_allowed}). "
                f"Each thread can use at most {self.gpu_params.max_registers_per_thread} registers, "
                f"with {cm.NUM_REG_PER_THREAD} reserved for kernel overhead."
            )

        # Calculate required registers per thread
        required_reg_per_thread = num_limbs + cm.NUM_REG_PER_THREAD

        if required_reg_per_thread > self.gpu_params.max_registers_per_thread:
            logger.error(f"Required registers {required_reg_per_thread} exceeds per-thread limit "
                        f"{self.gpu_params.max_registers_per_thread}")
            raise ValueError(
                f"Required registers per thread ({required_reg_per_thread}) exceeds "
                f"architectural limit ({self.gpu_params.max_registers_per_thread})"
            )

        # Compute max block size based on register constraints
        max_bs_register_limited = self.gpu_params.max_registers_per_sm // required_reg_per_thread

        # Also respect architectural max threads per block
        max_bs = min(max_bs_register_limited, self.gpu_params.max_threads_per_block)

        # Round down to nearest multiple of warp size for optimal performance
        max_bs = (max_bs // cm.WARP_SIZE) * cm.WARP_SIZE

        if max_bs == 0:
            logger.error(f"Computed max_block_size is 0 with num_limbs={num_limbs}")
            raise ValueError(
                f"Register pressure too high: cannot fit even one warp. "
                f"Required registers per thread: {required_reg_per_thread}"
            )

        logger.debug(f"Computed max_block_size={max_bs} for num_limbs={num_limbs} "
                    f"(registers/thread={required_reg_per_thread})")

        return max_bs

    def compute_dimensions(
        self,
        sample_size: int,
        num_limbs: int,
        enforce_warp_alignment: bool = True
    ) -> KernelDimensions:
        """Compute optimal kernel launch dimensions for the given workload.

        This method calculates grid size, block size, and number of iterations
        required to process all samples, considering GPU architecture constraints
        and register pressure.

        Strategy:
        1. If samples fit in one block: use minimal grid (1 block)
        2. If samples fit in one iteration across all SMs: distribute across multiple blocks
        3. Otherwise: use full GPU capacity and iterate multiple times

        Parameters
        ----------
        sample_size : int
            Total number of samples to process.
        num_limbs : int
            Number of limbs in the binary sequence per thread.
        enforce_warp_alignment : bool, optional
            If True, ensures block_size is a multiple of warp size. Default is True.

        Returns
        -------
        KernelDimensions
            Named tuple containing grid_size, block_size, and num_iterations.

        Raises
        ------
        ValueError
            If sample_size is non-positive or num_limbs is invalid.

        Examples
        --------
        >>> calculator = KernelDimensionCalculator()
        >>> dims = calculator.compute_dimensions(sample_size=1000000, num_limbs=128)
        >>> print(f"Launch config: grid={dims.grid_size}, block={dims.block_size}, iters={dims.num_iterations}")

        Notes
        -----
        For optimal performance, block sizes are aligned to warp boundaries (32 threads)
        to avoid warp divergence and maximize SM occupancy.
        """
        # Input validation
        if sample_size <= 0:
            logger.error(f"Invalid sample_size: {sample_size}. Must be positive.")
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        logger.info(f"Computing kernel dimensions for sample_size={sample_size}, num_limbs={num_limbs}")

        # Get maximum block size given register constraints
        max_bs = self.compute_max_block_size(num_limbs)

        # Maximum threads per iteration (full GPU utilization)
        max_total_threads_per_iter = max_bs * self.gpu_params.num_sm

        # Determine launch configuration
        if sample_size <= max_bs:
            # Case 1: All samples fit in a single block
            grid_size = 1
            block_size = sample_size
            num_iterations = 1

            logger.debug(f"Case 1: Single block - samples fit in one block")

        elif sample_size <= max_total_threads_per_iter:
            # Case 2: Multiple blocks needed, but fits in one iteration
            grid_size = math.ceil(sample_size / max_bs)
            block_size = max_bs
            num_iterations = 1

            logger.debug(f"Case 2: Multiple blocks - {grid_size} blocks needed")

        else:
            # Case 3: Multiple iterations required
            grid_size = self.gpu_params.num_sm
            block_size = max_bs
            num_iterations = math.ceil(sample_size / max_total_threads_per_iter)

            logger.debug(f"Case 3: Multiple iterations - {num_iterations} iterations needed")

        # Ensure block size is warp-aligned for optimal performance
        if enforce_warp_alignment and block_size % cm.WARP_SIZE != 0:
            aligned_block_size = (
                (block_size + cm.WARP_SIZE - 1) // cm.WARP_SIZE
            ) * cm.WARP_SIZE

            # Make sure we don't exceed max after alignment
            if aligned_block_size <= max_bs:
                logger.debug(f"Aligned block_size from {block_size} to {aligned_block_size}")
                block_size = aligned_block_size

        dimensions = KernelDimensions(
            grid_size=grid_size,
            block_size=block_size,
            num_iterations=num_iterations
        )

        total_threads_per_iter = grid_size * block_size
        logger.info(f"Computed dimensions: {dimensions}")
        logger.info(f"  Threads per iteration: {total_threads_per_iter:,}")
        logger.info(f"  Total thread launches: {total_threads_per_iter * num_iterations:,}")
        logger.info(f"  GPU utilization per iter: {(total_threads_per_iter / max_total_threads_per_iter) * 100:.1f}%")

        return dimensions

# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("GPU Kernel Dimension Calculator - Example Usage")
    print("=" * 80)

    try:
        # Initialize GPU parameters (auto-detects current device)
        gpu_params = GPUArchitectureParams()
        print(f"\n{gpu_params}")
        print(f"Device Name: {gpu_params.device_name}")
        print(f"Number of SMs: {gpu_params.num_sm}")
        print(f"Max Threads per Block: {gpu_params.max_threads_per_block}")
        print(f"Max Registers per SM: {gpu_params.max_registers_per_sm}")
        print(f"Warp Size: {cm.WARP_SIZE}")

        # Create calculator
        calculator = KernelDimensionCalculator(gpu_params)

        # Test cases
        test_cases = [
            {"sample_size": 1024, "num_limbs": 64, "description": "Small workload"},
            {"sample_size": 100000, "num_limbs": 128, "description": "Medium workload"},
            {"sample_size": 10000000, "num_limbs": 64, "description": "Large workload"},
            {"sample_size": 500, "num_limbs": 32, "description": "Tiny workload"},
        ]

        print("\n" + "=" * 80)
        print("Test Cases")
        print("=" * 80)

        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['description']}")
            print(f"  Sample Size: {test['sample_size']:,}")
            print(f"  Num Limbs: {test['num_limbs']}")

            dims = calculator.compute_dimensions(
                sample_size=test['sample_size'],
                num_limbs=test['num_limbs']
            )

            print(f"  → Grid Size: {dims.grid_size}")
            print(f"  → Block Size: {dims.block_size}")
            print(f"  → Iterations: {dims.num_iterations}")
            print(f"  → Total Threads: {dims.grid_size * dims.block_size * dims.num_iterations:,}")

        # Performance comparison
        print("\n" + "=" * 80)
        print("Register Pressure Analysis")
        print("=" * 80)

        sample_size = 1000000
        limb_sizes = [32, 64, 128, 192]

        print(f"\nFixed sample size: {sample_size:,}")
        print(f"{'Limbs':<10} {'Max Block Size':<16} {'Grid Size':<10} {'Block Size':<10} {'Iters':<8} {'Threads/Iter':<15}")
        print("-" * 80)

        for limbs in limb_sizes:
            try:
                dims = calculator.compute_dimensions(sample_size, limbs)
                threads_per_iter = dims.grid_size * dims.block_size
                print(f"{limbs:<10} {calculator.compute_max_block_size(limbs):<16} "
                      f"{dims.grid_size:<10} {dims.block_size:<10} {dims.num_iterations:<8} "
                      f"{threads_per_iter:<15,}")
            except ValueError as e:
                print(f"{limbs:<10} Error: {e}")

        print("\n" + "=" * 80)
        print("Validation Complete")
        print("=" * 80)

    except RuntimeError as e:
        print(f"\nError: {e}")
        print("This example requires a CUDA-capable GPU.")
    except Exception as e:
        logger.exception("Unexpected error during example execution")
        print(f"\nUnexpected error: {e}")
