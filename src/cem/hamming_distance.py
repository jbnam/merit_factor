"""
Hamming Distance Module
=======================

Hamming distance between two binary sequences represented
by a torch.Tensor of the same len_bin_seq computing module.

This module provides:
- Computation of the Hamming distance

Author: MJMARI
License: Apache License 2.0
Version: 0.1.0
"""
import logging

import torch
import numpy as np

import src.common as cm

# Initialize logger
logger = logging.getLogger(__name__)

# ============================================================================
# Mask depending on the length of binary sequences generating function
# ============================================================================

def hd_mask(len_bin_seq: int) -> np.uint32:
    """
    Create the 32-bit mask in which the first x - (((x-1) % y) + 1) bits
    are 1 and the rest bits are 0 for & operation, where x = len_bin_seq
    and y= cm.LIMB_BIT_SIZE.

    Parameters
    ----------
    len_bin_seq : int
        The length of binary sequences > 0.

    Returns
    -------
    A 32-bit mask to set the last ((x-1) % y) + 1 bits as 0 : np.uint.

    Examples
    --------
    >>> len_bin_seq = 119
    >>> hd_mask(len_bin_seq)
    """
    if len_bin_seq < 1:
        logger.error(f"The length of binary sequences must be at least 1: {len_bin_seq}")
        raise ValueError(f"The length of binary sequences must be at least 1: {len_bin_seq}")

    shift = (cm.LIMB_BIT_SIZE - (((len_bin_seq -1) % cm.LIMB_BIT_SIZE) + 1))

    return np.uint32(np.uint32(cm.MAX_UINT32) << shift)

# ============================================================================
# Hamming distance computing function
# ============================================================================

def hamming_distance(a: torch.Tensor,
                     b: torch.Tensor,
                     len_bin_seq: int,
                     np_mask: np.uint32
                    ) -> int:
    """
    Compute the Hamming distance between two 1-D tensors of integers of
    length num_limbs representing len_bin_seq bit binary sequences.

    Warning: This is a function performed on CPU not GPU.

    Parameters
    ----------
    a : torch.Tensor
        A binary sequence of length len_bin_seq.
    b : torch.Tensor
        A binary sequence of length len_bin_seq.
    len_bin_seq : int
        The length of binary sequences.
    np_mask : np.uint32
        Mask for popcount().

    Returns
    -------
    The Hamming distance between a and b as int.

    Examples
    --------
    >>> len_bin_seq = 119
    >>> np_mask = hd_mask(len_bin_seq)
    >>> a = torch.randint(high=2**32-1, size=(4,))
    >>> b = torch.randint(high=2**32-1, size=(4,))
    >>> hamming_distance(a, b, 119, np_mask)
    """
    # Check whether the lengths of a and b are same or not
    if len(a) != len(b):
        logger.error(f"The lengths of the input 1-D tensors do not match: {len(a)} and {len(b)}")
        raise ValueError(f"The lengths of the input 1-D tensors do not match: {len(a)} and {len(b)}")

    # Check whether the dimensions of a and b are 1 or not
    if a.dim() != 1 or b.dim() != 1:
        logger.error(f"The dimensions should be one: {a.dim()} for a and {b.dim()} for b")
        raise ValueError(f"The dimensions should be one: {a.dim()} for a and {b.dim()} for b")

    # Compute the Hamming distance by using XOR and
    hd = 0
    for j in range(len(a)-1):
        tensor_xor = torch.bitwise_xor(a[j], b[j])
        hd += np.bitwise_count(tensor_xor.numpy())
    tensor_xor = torch.bitwise_xor(a[len(a)-1], b[len(b)-1])
    hd += np.bitwise_count(tensor_xor.numpy() & np_mask)

    return (len_bin_seq - hd)

if __name__ == "__main__":
    """Comprehensive example demonstrating computing Hamming distances."""

    print("\n" + "=" * 80)
    print("CEM Golay Merit Factor - Hamming distance Examples")
    print("=" * 80)

    try:
        # Example 1: Basic Hamming distance computing example for len_bin_seq <= 32
        print("\n[Example 1] Hamming distance computing example for len_bin_seq <= 32")
        print("=" * 80)

        # Set the length of binary sequences
        len_bin_seq = 25

        # Compute the mask
        np_mask = hd_mask(len_bin_seq)

        # Generate two random 32-bit integers
        a = torch.randint(low=-1000000, high=1000000, size=(1,))
        b = torch.randint(low=-1000000, high=1000000, size=(1,))

        # Compute Hamming distance
        hd = hamming_distance(a, b, len_bin_seq, np_mask)

        print(f"  The length of binary sequences is {len_bin_seq}")

        print(f"\n  The corresponding mask is {(np_mask.item() & cm.MAX_UINT32):032b}")

        print(f"\n  The bit representation of a is {(a.item() & cm.MAX_UINT32):032b}")
        print(f"  The bit representation of b is {(b.item() & cm.MAX_UINT32):032b}")

        print(f"\n  Hamming distance is {hd}.")

        # Example 2: Hamming distance computing example for len_bin_seq > 32
        print("\n[Example 1] Hamming distance computing example for len_bin_seq > 32")
        print("=" * 80)

        # Set the length of binary sequences
        len_bin_seq = 91

        # Compute the mask
        np_mask = hd_mask(len_bin_seq)

        # Generate two random 32-bit integers
        a = torch.randint(low=-1000000, high=1000000, size=(3,))
        b = torch.randint(low=-1000000, high=1000000, size=(3,))

        # Compute Hamming distance
        hd = hamming_distance(a, b, len_bin_seq, np_mask)

        print(f"  The length of binary sequences is {len_bin_seq}")

        print(f"\n  The corresponding mask is {(np_mask.item() & cm.MAX_UINT32):032b}")

        print(f"\n  The bit representation of a[0] is {(a[0].item() & cm.MAX_UINT32):032b}")
        print(f"  The bit representation of a[1] is {(a[1].item() & cm.MAX_UINT32):032b}")
        print(f"  The bit representation of a[2] is {(a[2].item() & cm.MAX_UINT32):032b}")

        print(f"\n  The bit representation of b[0] is {(b[0].item() & cm.MAX_UINT32):032b}")
        print(f"  The bit representation of b[1] is {(b[1].item() & cm.MAX_UINT32):032b}")
        print(f"  The bit representation of b[2] is {(b[2].item() & cm.MAX_UINT32):032b}")

        print(f"\n  Hamming distance is {hd}.")

        print("\n" + "=" * 80)
        print(" All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n Error during example: {e}\n")
        raise
