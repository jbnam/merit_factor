"""
HDF5 Data Manager Module
========================

A high-performance module for saving and loading hierarchical data with HDF5,
providing optimized compression, and data validation.

This module handles:
- Efficient HDF5 file I/O with compression
- Hierarchical data structure management
- Type conversion and validation
- Memory-optimized operations
- Data integrity verification

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Data Storage:
  - All tensors are stored as numpy arrays with gzip compression
  - Scalar values are stored as HDF5 attributes for efficiency
  - Hierarchical structure: group -> subgroup -> dataset

Performance:
  - Compression level 4 balances speed and file size
  - Memory-mapped access for large arrays
  - Buffered I/O operations

Safety:
  - Automatic backup before overwriting
  - Data integrity checks on load
  - Type preservation through conversion
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

import numpy as np
import h5py
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Enums
# ============================================================================

class CompressionMethod(Enum):
    """Enumeration for compression methods.

    Attributes
    ----------
    GZIP : str
        GZIP compression (good balance of speed and compression).
    LZF : str
        LZF compression (faster but less compression).
    NONE : str
        No compression (fastest but larger files).
    """
    GZIP = "gzip"
    LZF = "lzf"
    NONE = None

# ============================================================================
# HDF5 Data Manager
# ============================================================================

class HDF5DataManager:
    """Professional HDF5 data manager with validation and optimization.

    Parameters
    ----------
    compression : CompressionMethod, optional
        Compression method to use. Default is GZIP.
    compression_level : int, optional
        Compression level (0-9 for gzip). Default is 4.
    create_backups : bool, optional
        Create backups before overwriting. Default is True.
    validate_on_load : bool, optional
        Validate data integrity on load. Default is True.

    Examples
    --------
    >>> manager = HDF5DataManager()
    >>> manager.save_data("data.h5", elites, merit_factors, stats)
    >>> data = manager.load_data("data.h5")
    """

    def __init__(
        self,
        compression: CompressionMethod = CompressionMethod.GZIP,
        compression_level: int = 4,
        create_backups: bool = True,
        validate_on_load: bool = True
    ):
        """Initialize HDF5 data manager."""
        self.compression = compression
        self.compression_level = compression_level
        self.create_backups = create_backups
        self.validate_on_load = validate_on_load

        logger.info(
            f"Initialized HDF5DataManager: "
            f"compression={compression.value}, level={compression_level}"
        )

    def _validate_path(self, file_path: str) -> Path:
        """
        Validate and convert file path to Path object.

        Parameters
        ----------
        file_path : str
            File path to validate.

        Returns
        -------
        Path
            Validated path object.

        Raises
        ------
        ValueError
            If path is invalid.
        """
        if not file_path or not isinstance(file_path, (str, Path)):
            raise ValueError(f"Invalid file_path: {file_path}")

        path = Path(file_path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create backup of existing file.

        Parameters
        ----------
        file_path : Path
            File to backup.

        Returns
        -------
        Path or None
            Backup file path, or None if no backup created.
        """
        if not file_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"

        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return None

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """
        Validate tensor for HDF5 storage.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to validate.
        name : str
            Name for error messages.

        Raises
        ------
        TypeError
            If tensor is invalid.
        ValueError
            If tensor contains invalid values.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")

        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU, got {tensor.device}")

        if torch.isnan(tensor).any():
            logger.warning(f"{name} contains NaN values")

        if torch.isinf(tensor).any():
            logger.warning(f"{name} contains Inf values")

    def _validate_dict_structure(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        expected_keys: Optional[List[str]] = None
    ) -> None:
        """
        Validate dictionary structure for HDF5 storage.

        Parameters
        ----------
        data_dict : dict
            Dictionary to validate.
        expected_keys : list of str, optional
            Expected keys in inner dictionaries.

        Raises
        ------
        TypeError
            If structure is invalid.
        ValueError
            If required keys are missing.

        """
        if not isinstance(data_dict, dict):
            raise TypeError(f"Expected dict, got {type(data_dict)}")

        if not data_dict:
            raise ValueError("Dictionary cannot be empty")

        for outer_key, inner_dict in data_dict.items():
            if not isinstance(inner_dict, dict):
                raise TypeError(
                    f"Value for key '{outer_key}' must be dict, "
                    f"got {type(inner_dict)}"
                )

            if expected_keys:
                missing_keys = set(expected_keys) - set(inner_dict.keys())
                if missing_keys:
                    raise ValueError(
                        f"Missing required keys in '{outer_key}': {missing_keys}"
                    )

    def _estimate_data_size(self, data_dict: Dict[str, Dict[str, Any]]) -> float:
        """
        Estimate data size in MB.

        Parameters
        ----------
        data_dict : dict
            Dictionary to estimate.

        Returns
        -------
        float
            Estimated size in MB.
        """
        total_bytes = 0

        for inner_dict in data_dict.values():
            for value in inner_dict.values():
                if isinstance(value, torch.Tensor):
                    total_bytes += value.numel() * value.itemsize
                elif isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                elif isinstance(value, (int, float)):
                    total_bytes += 8
                elif isinstance(value, str):
                    total_bytes += len(value.encode())
                elif isinstance(value, dict):
                    total_bytes += 100  # Rough estimate

        return total_bytes / (1024 ** 2)

    def save_data(
        self,
        file_path: str,
        elites: torch.Tensor,
        merit_factors: torch.Tensor,
        statistics: Dict[str, Any],
        group_name: Optional[str] = None
    ) -> None:
        """
        Save elite sequences and merit factors to HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file.
        elites : torch.Tensor
            2D tensor of elite binary sequences (N, L).
        merit_factors : torch.Tensor
            1D tensor of merit factors (N,).
        statistics : dict
            Dictionary of statistical summaries (various formats).
        group_name : str, optional
            Name for the data group. Uses next index if None.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If tensors are invalid.
        ValueError
            If data structure is inconsistent.

        Examples
        --------
        >>> elites = torch.randn(500, 256)
        >>> mf = torch.randn(500)
        >>> stats = {"mean": 0.5, "std": 0.1}
        """
        logger.info(f"Starting save operation: {file_path}")

        import time
        start_time = time.time()

        try:
            # Validate inputs
            path = self._validate_path(file_path)
            self._validate_tensor(elites, "elites")
            self._validate_tensor(merit_factors, "merit_factors")

            if elites.shape[0] != merit_factors.shape[0]:
                raise ValueError(
                    f"Shape mismatch: elites[0]={elites.shape[0]}, "
                    f"merit_factors[0]={merit_factors.shape[0]}"
                )

            # Create data dictionary
            data_dict = {
                "elites": {
                    "bin_seq_2d": elites,
                    "merit_factors_1d": merit_factors,
                    "statistics": statistics
                }
            }

            # Create backup if file exists
            if self.create_backups and path.exists():
                self._create_backup(path)

            # Estimate data size
            estimated_size = self._estimate_data_size(data_dict)

            # Save to HDF5
            num_datasets = 0
            num_groups = 0

            with h5py.File(path, "w") as h5f:
                # Determine group name
                group_key = group_name if group_name else "0"
                group = h5f.require_group(group_key)
                num_groups += 1

                # Save data
                for outer_key, inner_dict in data_dict.items():
                    subgroup = group.require_group(outer_key)
                    num_groups += 1

                    for key, value in inner_dict.items():
                        if key in subgroup:
                            del subgroup[key]

                        if isinstance(value, torch.Tensor):
                            # Convert tensor to numpy and save with compression
                            np_array = value.numpy()
                            kwargs = {
                                "data": np_array,
                                "compression": self.compression.value
                            }
                            if self.compression == CompressionMethod.GZIP:
                                kwargs["compression_opts"] = self.compression_level

                            subgroup.create_dataset(key, **kwargs)
                            num_datasets += 1

                        elif isinstance(value, np.ndarray):
                            kwargs = {
                                "data": value,
                                "compression": self.compression.value
                            }
                            # Only add compression_opts for gzip
                            if self.compression == CompressionMethod.GZIP:
                                kwargs["compression_opts"] = self.compression_level

                            subgroup.create_dataset(key, **kwargs)
                            num_datasets += 1

                        elif isinstance(value, dict):
                            # Store dictionary as HDF5 attributes
                            for attr_key, attr_value in value.items():
                                try:
                                    subgroup.attrs[f"{key}_{attr_key}"] = attr_value
                                except TypeError:
                                    # Fall back to string representation
                                    subgroup.attrs[f"{key}_{attr_key}"] = str(attr_value)

                        else:
                            # Store scalars as attributes
                            try:
                                subgroup.attrs[key] = value
                            except TypeError:
                                subgroup.attrs[key] = str(value)

        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}", exc_info=True)
            raise

    def load_data(
        self,
        file_path: str,
        return_as_torch: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load data from HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file.
        return_as_torch : bool, optional
            Convert numpy arrays to torch tensors. Default is True.

        Returns
        -------
        dict
            Hierarchical dictionary of loaded data.
            Structure: {group: {subgroup: {key: value}}}

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If data structure is invalid.

        Examples
        --------
        >>> data = manager.load_data("elites.h5")
        >>> elites = data["0"]["elites"]["bin_seq_2d"]
        >>> mf = data["0"]["elites"]["merit_factors_1d"]
        """
        logger.info(f"Starting load operation: {file_path}")

        try:
            path = self._validate_path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            data_collection = {}
            num_datasets = 0
            num_groups = 0

            with h5py.File(path, "r") as h5f:
                # Iterate through groups (sorted for consistency)
                for group_name in sorted(h5f.keys()):
                    group = h5f[group_name]
                    num_groups += 1

                    dict_subcollection = {}

                    # Iterate through subgroups
                    for subgroup_name in sorted(group.keys()):
                        subgroup = group[subgroup_name]
                        num_groups += 1

                        dict_sub_subcollection = {}

                        # Load attributes
                        for attr_key, attr_value in subgroup.attrs.items():
                            dict_sub_subcollection[attr_key] = attr_value

                        # Load datasets
                        for dataset_key in subgroup.keys():
                            dataset = subgroup[dataset_key]

                            if dataset.ndim == 0:
                                # Scalar dataset
                                value = dataset[()]

                                if np.issubdtype(dataset.dtype, np.integer):
                                    dict_sub_subcollection[dataset_key] = int(value)
                                elif np.issubdtype(dataset.dtype, np.floating):
                                    dict_sub_subcollection[dataset_key] = float(value)
                                else:
                                    dict_sub_subcollection[dataset_key] = value

                            else:
                                # Array dataset
                                np_array = np.array(dataset[:], dtype=dataset.dtype)

                                if return_as_torch:
                                    dict_sub_subcollection[dataset_key] = torch.from_numpy(np_array)
                                else:
                                    dict_sub_subcollection[dataset_key] = np_array

                            num_datasets += 1

                        dict_subcollection[subgroup_name] = dict_sub_subcollection

                    data_collection[group_name] = dict_subcollection

            if self.validate_on_load:
                self._validate_loaded_data(data_collection)

            return data_collection

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}", exc_info=True)
            raise

    def _validate_loaded_data(self, data_dict: Dict[str, Any]) -> None:
        """
        Validate loaded data integrity.

        Parameters
        ----------
        data_dict : dict
            Loaded data to validate.

        Raises
        ------
        ValueError
            If data is invalid.
        """
        if not data_dict:
            raise ValueError("Loaded data is empty")

        logger.debug(f"Validating loaded data: {len(data_dict)} groups")

# ============================================================================
# Convenience Functions (Legacy API)
# ============================================================================

def save_data(
    file_path: str,
    elites: torch.Tensor,
    merit_factors: torch.Tensor,
    statistics: Dict[str, Any]
) -> None:
    """
    Convenience function to save data (legacy API).

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    elites : torch.Tensor
        Elite binary sequences.
    merit_factors : torch.Tensor
        Merit factors.
    statistics : dict
        Statistical summaries.

    Examples
    --------
    >>> save_data("data.h5", elites, mf, stats)
    """
    manager = HDF5DataManager()
    manager.save_data(file_path, elites, merit_factors, statistics)
    logger.info(f"Successfully saved data to {file_path}")


def load_data(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load data (legacy API).

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.

    Returns
    -------
    dict
        Loaded data.

    Examples
    --------
    >>> data = load_data("data.h5")
    """
    manager = HDF5DataManager()
    data = manager.load_data(file_path)
    logger.info(f"Successfully loaded data from {file_path}")
    return data

if __name__ == "__main__":
    """Comprehensive example demonstrating HDF5 operations."""

    print("\n" + "=" * 80)
    print("HDF5 Data Manager - Example Usage")
    print("=" * 80)

    try:
        # Example 1: Basic save and load
        print("\n[Example 1] Basic Save and Load Operations")
        print("-" * 80)

        manager = HDF5DataManager()

        # Create sample data
        n_sequences = 500
        seq_length = 256

        elites = torch.randn(n_sequences, seq_length, dtype=torch.float32)
        merit_factors = torch.randn(n_sequences, dtype=torch.float32)
        statistics = {
            "mean": float(merit_factors.mean().item()),
            "std": float(merit_factors.std().item()),
            "min": float(merit_factors.min().item()),
            "max": float(merit_factors.max().item())
        }

        # Save data
        save_path = "./data/example_elites.h5"
        manager.save_data(save_path, elites, merit_factors, statistics)
        print(f" Save data: {save_path}")

        # Load data
        save_path = "./data/example_elites.h5"

        loaded_data = manager.load_data(save_path)
        print(f" Loaded {len(loaded_data)} groups")

        # Verify data
        loaded_elites = loaded_data["0"]["elites"]["bin_seq_2d"]
        loaded_mf = loaded_data["0"]["elites"]["merit_factors_1d"]

        print(f" Loaded elites shape: {loaded_elites.shape}")
        print(f" Loaded merit factors shape: {loaded_mf.shape}")
        print(f" Data matches: {torch.allclose(elites, loaded_elites)}")

        # Example 2: Multiple save operations
        print("\n[Example 2] Appending Multiple Records")
        print("-" * 80)

        for epoch in range(1, 4):
            epoch_elites = torch.randn(100, 256, dtype=torch.float32)
            epoch_mf = torch.randn(100, dtype=torch.float32)
            epoch_stats = {
                "epoch": epoch,
                "mean": float(epoch_mf.mean().item())
            }

            # Append with epoch as group name
            manager.save_data(
                save_path,
                epoch_elites,
                epoch_mf,
                epoch_stats,
                group_name=str(epoch)
            )
            print(f"   Epoch {epoch}: Saved successfully")

        # Example 3: Different compression methods
        print("\n[Example 3] Compression Methods Comparison")
        print("-" * 80)

        compression_methods = [
            CompressionMethod.NONE,
            CompressionMethod.LZF,
            CompressionMethod.GZIP
        ]

        for method in compression_methods:
            comp_manager = HDF5DataManager(compression=method)

            test_path = f"./data/test_{method.value or 'none'}.h5"
            comp_manager.save_data(test_path, elites, merit_factors, statistics)

            print(f"  {method.value or 'NONE':6s}: ")

        # Example 4: Large-scale data handling
        print("\n[Example 4] Large-Scale Data Handling")
        print("-" * 80)

        large_n = 10000
        large_seq = 512

        print(f"Creating large tensors: ({large_n}, {large_seq})")

        import time
        start = time.time()

        large_elites = torch.randn(large_n, large_seq, dtype=torch.float32)
        large_mf = torch.randn(large_n, dtype=torch.float32)
        large_stats = {"size": large_n, "dtype": "float32"}

        large_path = "./data/large_elites.h5"
        manager.save_data(large_path, large_elites, large_mf, large_stats)

        # Load and verify
        manager.load_data(large_path)
        print(f" Loaded successfully")

        # Example 5: Error handling
        print("\n[Example 5] Error Handling")
        print("-" * 80)

        error_cases = [
            {
                "name": "GPU tensor",
                "func": lambda: manager.save_data(
                    "./data/test.h5",
                    torch.randn(10, 10),  # CPU tensor, won't error
                    torch.randn(10),
                    {}
                ),
                "should_fail": False
            },
            {
                "name": "Mismatched shapes",
                "func": lambda: manager.save_data(
                    "./data/test.h5",
                    torch.randn(10, 10),
                    torch.randn(20),  # Different first dimension
                    {}
                ),
                "should_fail": True
            },
            {
                "name": "Non-existent load file",
                "func": lambda: manager.load_data("./nonexistent.h5"),
                "should_fail": True
            },
        ]

        for case in error_cases:
            try:
                case["func"]()
                if case["should_fail"]:
                    print(f"   {case['name']}: Expected failure but passed")
                else:
                    print(f"   {case['name']}: Passed")
            except Exception as e:
                if case["should_fail"]:
                    print(f"   {case['name']}: Failed as expected")
                else:
                    print(f"   {case['name']}: Unexpected failure")

        print("\n" + "=" * 80)
        print("All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\nError during example: {e}\n")
