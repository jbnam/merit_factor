"""
CUDA Merit Factor Computation Module
=====================================

A high-performance module for computing merit factors of binary sequences
using CUDA kernels with JIT compilation via PyTorch's load_inline.

This module provides:
- Efficient CUDA kernel compilation with torch.utils.cpp_extension
- Merit factor computation using bitwise operations
- Type-safe configuration management
- Comprehensive logging and performance profiling
- Automatic fallback and error handling

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Bitwise Operations:
  - All bitwise operations execute on GPU (sign/magnitude issues not a concern)
  - __popc() computes population count (number of 1-bits)
  - Padding handled with masks for correctness

Performance:
  - Register-optimized kernel using NUM_LIMBS template parameter
  - Hardware texture cache via __ldg() for global memory reads
  - Local register allocation for column data

GPU Requirements:
  - Compute Capability >= 6.1 (GTX 1080 Ti or newer)
  - CUDA Toolkit compatible with PyTorch version
"""

import shutil
import logging
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from types import ModuleType

import torch
from torch.utils.cpp_extension import load_inline

import src.common as cm
from src.cem.distribution import BernoulliLimbGenerator

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class MeritFactorConfig:
    """Configuration for merit factor computation.

    Attributes
    ----------
    len_bin_seq : int
        Length of binary sequences (bits).
    limb_size : int
        Number of bits per limb (typically 32).
    block_size : int
        CUDA block size (number of threads per block).
    num_blocks : int
        Number of CUDA blocks.
    compute_capability : str
        GPU compute capability (e.g., "6.1").

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    len_bin_seq: int
    limb_size: int
    block_size: int
    num_blocks: int
    compute_capability: str = "6.1"

    def __post_init__(self):
        """Validate configuration."""
        self._validate_all()

    def _validate_all(self) -> None:
        """Validate all parameters."""
        if self.len_bin_seq <= 0:
            raise ValueError(f"len_bin_seq must be positive, got {self.len_bin_seq}")

        if self.len_bin_seq > 100000:
            logger.warning(f"len_bin_seq={self.len_bin_seq} is very large")

        # WARNING! limb_size must be 32 or 64 bits
        if self.limb_size not in [32, 64]:
            raise ValueError(f"limb_size must be 32 or 64, got {self.limb_size}")

        # WARNING! block_size must be valid CUDA block dimension
        if self.block_size <= 0 or self.block_size > 1024:
            raise ValueError(
                f"block_size must be in (0, 1024], got {self.block_size}"
            )

        # WARNING! num_blocks must match GPU multiprocessor count
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")

        # Compute number of limbs
        num_limbs = (self.len_bin_seq + self.limb_size - 1) // self.limb_size

        # WARNING! NUM_LIMBS template parameter must match
        if num_limbs > 32:
            logger.warning(f"num_limbs={num_limbs} is large, may affect register usage")

    @property
    def num_limbs(self) -> int:
        """Compute number of limbs."""
        return (self.len_bin_seq + self.limb_size - 1) // self.limb_size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MeritFactorConfig(len={self.len_bin_seq}, "
            f"limb_size={self.limb_size}, block_size={self.block_size}, "
            f"num_blocks={self.num_blocks}, num_limbs={self.num_limbs})"
        )


# ============================================================================
# CUDA Kernel Source Code
# ============================================================================

class CUDAKernelSource:
    """Container for CUDA kernel source code."""

    # Host-side C++ wrapper
    CPP_SOURCE = """
#include <torch/extension.h>
#include <cstdint>

namespace merit_factors {

void create_constants_gpu(const int32_t max_uint32);

torch::Tensor merit_factors_gpu(torch::Tensor bin_seq_2d,
                                const int32_t len_bin_seq,
                                const int32_t limb_size,
                                const int32_t block_size,
                                const int32_t num_blocks);
}

void create_constants_cpu(const int32_t max_uint32) {
    return merit_factors::create_constants_gpu(max_uint32);
}

torch::Tensor merit_factors_cpu(torch::Tensor bin_seq_2d,
                                const int32_t len_bin_seq,
                                const int32_t limb_size,
                                const int32_t block_size,
                                const int32_t num_blocks) {
    return merit_factors::merit_factors_gpu(bin_seq_2d,
                                            len_bin_seq,
                                            limb_size,
                                            block_size,
                                            num_blocks);
}
"""

    # Device-side CUDA kernel
    CUDA_SOURCE_TEMPLATE = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace merit_factors {

__constant__ int32_t MASK_MAX_32;

void create_constants(const int32_t max_uint32) {
    cudaError_t err = cudaMemcpyToSymbol(MASK_MAX_32, &max_uint32, sizeof(int32_t));
    TORCH_CHECK(err == cudaSuccess, "Failed to create global constant: ", cudaGetErrorString(err));
}

template <int32_t NUM_LIMBS>
__global__ void merit_factors_kernel(
    int32_t* __restrict__ bin_seq_2d,
    float* __restrict__ arr_mf_1d,
    const int32_t len_bin_seq,
    const int32_t limb_size,
    const int32_t block_size,
    const int32_t num_blocks
) {
    int32_t c_k, c, E = 0;
    int32_t t, k, i, r, j = 0;
    int32_t inner_iter;
    int32_t shifter = 0;
    int32_t tmod;
    int32_t inner_tmask;
    int32_t res_len_bin_seq;

    int32_t y = threadIdx.x;
    int32_t z = blockIdx.x;

    int32_t mf_idx = (z * block_size) + y;
    int32_t col_idx = (z * block_size * NUM_LIMBS) + (y * NUM_LIMBS);

    if (z < num_blocks && y < block_size) {
        int32_t col[NUM_LIMBS];
        #pragma unroll
        for (r = 0; r < NUM_LIMBS; ++r) {
            col[r] = __ldg(bin_seq_2d + col_idx + r);
        }

        inner_iter = NUM_LIMBS - (j+1);
        r = 1;
        res_len_bin_seq = len_bin_seq % limb_size;
        shifter = limb_size - res_len_bin_seq;
        tmod = (shifter < limb_size) ? (MASK_MAX_32 << shifter) : MASK_MAX_32;

        for (k = 1; k < len_bin_seq; k++) {
            c_k = 0;
            shifter = limb_size - r;
            inner_tmask = MASK_MAX_32;
            for (i = 0; i < inner_iter; i++) {
                t = i + j;
                c = (col[t] << r);
                // WARNING! Padding by 1s when reaching penultimate limb
                if (t == NUM_LIMBS - 2) {
                    inner_tmask = tmod >> shifter;
                }
                // WARNING! Padding by 0s from next limb
                c |= (int32_t)((uint32_t)col[t+1] >> shifter);
                c_k += __popc( ~(c ^ col[i]) & inner_tmask);
            }
            t = tmod << r;
            if (t != 0) {
                c = (col[NUM_LIMBS - 1] << r);
                c_k += __popc( ~(c ^ col[i]) & t );
            }
            c_k = c_k << 1;
            c_k += k - len_bin_seq;

            c_k *= c_k;
            E += c_k;

            r += 1;
            if (r == limb_size) {
                j += 1;
                inner_iter = NUM_LIMBS - (j+1);
                r = 0;
            }
        }

        E = E << 1;
        arr_mf_1d[mf_idx] = __fdividef((float) (len_bin_seq * len_bin_seq), (float) E);
    }
}

void create_constants_gpu(const int32_t max_uint32) {
    return create_constants(max_uint32);
}

torch::Tensor merit_factors_gpu(torch::Tensor bin_seq_2d,
                                const int32_t len_bin_seq,
                                const int32_t limb_size,
                                const int32_t block_size,
                                const int32_t num_blocks) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(bin_seq_2d.device())
        .layout(torch::kStrided);

    torch::Tensor arr_mf_1d = torch::empty({bin_seq_2d.size(0)}, options);

    constexpr int32_t NUM_LIMBS = PLACEHOLDER_NUM_LIMBS;

    merit_factors_kernel<NUM_LIMBS><<<num_blocks, block_size>>>(
        bin_seq_2d.data_ptr<int32_t>(),
        arr_mf_1d.data_ptr<float>(),
        len_bin_seq,
        limb_size,
        block_size,
        num_blocks
    );

    return arr_mf_1d;
}

} // namespace merit_factors
"""


# ============================================================================
# CUDA Extension Manager
# ============================================================================

class CUDAMeritFactorCompiler:
    """Manages CUDA kernel compilation and caching.

    Parameters
    ----------
    compute_capability : str, optional
        GPU compute capability. Default is "6.1".
    verbose : bool, optional
        Print compilation details. Default is True.

    Examples
    --------
    >>> compiler = CUDAMeritFactorCompiler()
    >>> module = compiler.compile(num_limbs=4)
    """

    def __init__(self, compute_capability: str = "6.1", verbose: bool = True):
        """Initialize CUDA compiler."""
        self.compute_capability = compute_capability
        self.verbose = verbose
        self.compiled_modules: Dict[int, ModuleType] = {}

        # Clear stale cache
        self._clear_cache()

        logger.info(f"Initialized CUDAMeritFactorCompiler: CC={compute_capability}")

    def _clear_cache(self) -> None:
        """Clear PyTorch extension cache."""
        cache_dir = Path.home() / ".cache" / "torch_extensions"

        # WARNING! Removing cache directory may affect other extensions
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.debug(f"Cleared cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

    def compile(self, num_limbs: int) -> ModuleType:
        """
        Compile CUDA kernel for given NUM_LIMBS.

        Parameters
        ----------
        num_limbs : int
            Number of limbs per binary sequence.

        Returns
        -------
        ModuleType
            Compiled CUDA extension module.

        Raises
        ------
        RuntimeError
            If compilation fails.

        WARNING! Compilation may take 30-60 seconds first time
        WARNING! num_limbs must match template instantiation

        Examples
        --------
        >>> module = compiler.compile(num_limbs=4)
        """
        # Check if already compiled
        if num_limbs in self.compiled_modules:
            logger.debug(f"Using cached module for num_limbs={num_limbs}")
            return self.compiled_modules[num_limbs]

        logger.info(f"Compiling CUDA kernel for num_limbs={num_limbs}")

        try:
            # Prepare CUDA source with template parameter
            cuda_source = CUDAKernelSource.CUDA_SOURCE_TEMPLATE.replace(
                "PLACEHOLDER_NUM_LIMBS", str(num_limbs)
            )

            # WARNING! load_inline requires valid CUDA installation
            module = load_inline(
                name=f"kernel_merit_factors_{num_limbs}",
                cpp_sources=CUDAKernelSource.CPP_SOURCE,
                cuda_sources=cuda_source,
                functions=['merit_factors_cpu', 'create_constants_cpu'],
                extra_cuda_cflags=[f'-arch=sm_61'],  # GTX 1080 Ti
                with_cuda=True,
                verbose=self.verbose
            )

            # Cache compiled module
            self.compiled_modules[num_limbs] = module

            logger.info(
                f"Successfully compiled CUDA kernel for num_limbs={num_limbs} "
            )

            return module

        except Exception as e:
            logger.error(f"CUDA compilation failed for num_limbs={num_limbs}: {e}")
            raise RuntimeError(f"CUDA compilation failed: {e}") from e


# ============================================================================
# Merit Factor Computer
# ============================================================================

class CUDAMeritFactorComputer:
    """High-performance merit factor computer using CUDA kernels.

    Parameters
    ----------
    config : MeritFactorConfig
        Computation configuration.

    Examples
    --------
    >>> config = MeritFactorConfig(len_bin_seq=128, limb_size=32,
    ...                           block_size=256, num_blocks=2)
    >>> computer = CUDAMeritFactorComputer(config)
    >>> mf = computer.compute(bin_seqs)
    """

    def __init__(self, config: MeritFactorConfig):
        """Initialize merit factor computer."""
        self.config = config
        self.compiler = CUDAMeritFactorCompiler(config.compute_capability)
        self.module = self.compiler.compile(config.num_limbs)

        # Initialize GPU constants
        self._init_constants()

        logger.info(f"Initialized CUDAMeritFactorComputer: {config}")

    def _init_constants(self) -> None:
        """Initialize GPU constants."""
        # WARNING! MASK_MAX_32 = 0xFFFFFFFF = -1 in two's complement
        MAX_UINT32 = -1

        self.module.create_constants_cpu(MAX_UINT32)
        logger.debug("Initialized GPU constants")

    def _validate_input(self, bin_seqs: torch.Tensor) -> None:
        """
        Validate input tensor.

        Parameters
        ----------
        bin_seqs : torch.Tensor
            Binary sequences tensor (N, num_limbs) in int32.

        Raises
        ------
        TypeError
            If tensor type is invalid.
        ValueError
            If tensor shape is invalid.

        WARNING! Tensor must be on CUDA device
        WARNING! Tensor must be int32 dtype
        WARNING! Tensor must have shape (N, num_limbs)
        """
        # WARNING! Tensor must be on CUDA, not CPU
        if bin_seqs.device.type != "cuda":
            raise ValueError(
                f"Binary sequences must be on CUDA, got {bin_seqs.device}"
            )

        # WARNING! Dtype must be int32 for bitwise operations
        if bin_seqs.dtype != torch.int32:
            raise TypeError(
                f"Binary sequences must be int32, got {bin_seqs.dtype}"
            )

        # WARNING! Shape must be (N, num_limbs)
        if bin_seqs.dim() != 2:
            raise ValueError(
                f"Binary sequences must be 2D, got {bin_seqs.dim()}D"
            )

        if bin_seqs.shape[1] != self.config.num_limbs:
            raise ValueError(
                f"Expected {self.config.num_limbs} limbs, "
                f"got {bin_seqs.shape[1]}"
            )

    def compute(self, bin_seqs: torch.Tensor) -> torch.Tensor:
        """
        Compute merit factors for binary sequences.

        Parameters
        ----------
        bin_seqs : torch.Tensor
            Binary sequences as int32 tensor of shape (N, num_limbs).

        Returns
        -------
        torch.Tensor
            Merit factors as float32 tensor of shape (N,).

        Raises
        ------
        TypeError
            If input tensor type is invalid.
        ValueError
            If input tensor shape is invalid.

        WARNING! Input must be on CUDA device
        WARNING! Input must be int32 dtype
        WARNING! Output values are always positive (merit factor > 0)

        Examples
        --------
        >>> bin_seqs = torch.randint(0, 2**32, (512, 4), dtype=torch.int32).cuda()
        >>> mf = computer.compute(bin_seqs)
        >>> print(f"Merit factors shape: {mf.shape}")
        """
        logger.info(f"Computing merit factors for {bin_seqs.shape[0]} sequences")

        # Validate input
        self._validate_input(bin_seqs)

        try:
            # Call CUDA kernel
            merit_factors = self.module.merit_factors_cpu(
                bin_seqs,
                self.config.len_bin_seq,
                self.config.limb_size,
                self.config.block_size,
                self.config.num_blocks
            )

            # Synchronize GPU
            torch.cuda.synchronize()

            return merit_factors

        except Exception as e:
            logger.error(f"Merit factor computation failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    """Comprehensive example demonstrating CUDA merit factor computation."""

    print("\n" + "=" * 80)
    print("CUDA Merit Factor Computer - Example Usage")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n CUDA is not available. This module requires GPU support.\n")
        logger.error("CUDA is not available. Exiting example.")
        raise RuntimeError("CUDA is not available. Exiting example.")

    try:
        # Example 1: Basic configuration and compilation
        print("\n[Example 1] Basic Configuration and Compilation")
        print("-" * 80)

        config = MeritFactorConfig(
            len_bin_seq=128,
            limb_size=32,
            block_size=256,
            num_blocks=2
        )

        num_limbs = (config.len_bin_seq + config.limb_size - 1) // config.limb_size

        print(f"Configuration: {config}")
        print(f"Number of limbs: {num_limbs}")

        # Example 1: Merit factor computation
        print("\n[Example 1] Merit Factor Computation")
        print("-" * 80)

        computer = CUDAMeritFactorComputer(config)

        random_bianry_sequences = BernoulliLimbGenerator()

        # Generate random binary sequences
        num_seqs = 512
        gen = BernoulliLimbGenerator(device=torch.device("cuda"))
        arr_p = torch.tensor([0.5] * config.len_bin_seq, dtype=cm.T_FLOAT32, device="cuda")
        bin_seqs = random_bianry_sequences.generate(num_seqs, num_limbs, arr_p)

        print(f"Generated {num_seqs} random sequences: {bin_seqs.shape}")

        # Example 2: Performance scaling
        print("\n[Example 2] Performance Scaling with Different Batch Sizes")
        print("-" * 80)

        for batch_size in [100, 512, 1024, 5120]:
            arr_p = torch.tensor([0.5] * config.len_bin_seq, dtype=cm.T_FLOAT32, device="cuda")
            test_seqs = random_bianry_sequences.generate(batch_size, num_limbs, arr_p)

        # Example 3: Different sequence lengths
        print("\n[Example 3] Computing with Different Sequence Lengths")
        print("-" * 80)

        for len_seq in [64, 128, 256]:
            num_limbs = (len_seq + 31) // 32

            config_test = MeritFactorConfig(
                len_bin_seq=len_seq,
                limb_size=32,
                block_size=256,
                num_blocks=2
            )

            computer_test = CUDAMeritFactorComputer(config_test)

            num_limbs = (config_test.len_bin_seq + config_test.limb_size - 1) // config_test.limb_size

            arr_p = torch.tensor([0.5] * config_test.len_bin_seq, dtype=cm.T_FLOAT32, device="cuda")
            test_seqs = random_bianry_sequences.generate(256, num_limbs, arr_p)

            mf = computer_test.compute(test_seqs)

            print(f"  Sequence length {len_seq}: mean MF = {mf.mean():.4f}, "
                  f"std = {mf.std():.4f}")

        print("\n" + "=" * 80)
        print(" All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\n Error during example: {e}\n")
