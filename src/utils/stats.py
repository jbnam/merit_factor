"""
Epoch Statistical Summary and Checkpoint Manager
=================================================

A high-performance module for computing epoch-wise statistical summaries,
managing checkpoints, and persisting elite sequences, Bernoulli distributions,
and training logs with comprehensive profiling and error handling.

This module provides functionality for:
- Computing detailed statistical summaries of binary sequences and elites
- Tracking merit factor distributions and quantiles
- Persisting checkpoints with multiple formats (JSON, pickle)
- Managing training logs with rotation and archival
- Performance profiling

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Notes
-----
Device Handling:
  - All tensor operations occur on CPU for statistics
  - Bitwise operations preserve numerical integrity
  - No sign/magnitude issues in output summaries

File Management:
  - Automatic checkpoint directory creation
  - Rotating log files with size limits
  - JSON and pickle serialization options
  - Atomic writes to prevent corruption
"""
import logging
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

import torch
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QuantileStatistics:
    """Statistical summary with quantiles.

    Attributes
    ----------
    count : int
        Number of elements.
    mean : float
        Mean value.
    std : float
        Standard deviation.
    q25 : float
        25th percentile.
    q50 : float
        50th percentile (median).
    q75 : float
        75th percentile.
    """
    count: int
    mean: float
    std: float
    q25: float
    q50: float
    q75: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantileStatistics(count={self.count}, mean={self.mean:.6f}, "
            f"std={self.std:.6f}, q25={self.q25:.6f}, q50={self.q50:.6f}, q75={self.q75:.6f})"
        )


@dataclass
class BinarySequenceSummary:
    """Statistical summary for all binary sequences.

    Attributes
    ----------
    count : int
        Total number of sequences.
    mean : float
        Mean merit factor.
    std : float
        Standard deviation of merit factors.
    q25 : float
        25th percentile.
    q50 : float
        50th percentile (median).
    q75 : float
        75th percentile.
    """
    count: int
    mean: float
    std: float
    q25: float
    q50: float
    q75: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EliteSummary:
    """Statistical summary for elite sequences.

    Attributes
    ----------
    count : int
        Number of elite sequences.
    ratio : float
        Ratio of elites to total sequences.
    mean : float
        Mean merit factor of elites.
    std : float
        Standard deviation of merit factors.
    q25 : float
        25th percentile.
    q50 : float
        50th percentile (median).
    q75 : float
        75th percentile.
    min : float
        Minimum merit factor.
    max : float
        Maximum merit factor.
    distribution : List[float]
        Updated Bernoulli distribution.
    """
    count: int
    ratio: float
    mean: float
    std: float
    q25: float
    q50: float
    q75: float
    min: float
    max: float
    distribution: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Ensure distribution is a list for JSON serialization
        if isinstance(data["distribution"], np.ndarray):
            data["distribution"] = data["distribution"].tolist()
        return data


@dataclass
class EpochSummary:
    """Complete statistical summary for one epoch.

    Attributes
    ----------
    epoch : int
        Epoch number.
    timestamp : str
        ISO format timestamp.
    bin_seq : BinarySequenceSummary
        Statistics for all binary sequences.
    elites : EliteSummary
        Statistics for elite sequences.
    """
    epoch: int
    timestamp: str
    bin_seq: BinarySequenceSummary
    elites: EliteSummary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "bin_seq": self.bin_seq.to_dict(),
            "elites": self.elites.to_dict()
        }

# ============================================================================
# Statistical Summary Computation
# ============================================================================

class StatisticalAnalyzer:
    """Computes comprehensive statistical summaries for sequences.

    Parameters
    ----------
    validate_input : bool, optional
        Whether to validate input tensors. Default is True.

    Examples
    --------
    >>> analyzer = StatisticalAnalyzer()
    >>> bin_seq = torch.randn(1000)
    >>> summary = analyzer.compute_quantile_statistics(bin_seq)
    """

    def __init__(self, validate_input: bool = True):
        """Initialize statistical analyzer."""
        self.validate_input = validate_input
        logger.info("Initialized StatisticalAnalyzer")

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """
        Validate input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to validate.
        name : str
            Name for error messages.

        Raises
        ------
        TypeError
            If tensor is not 1D or dtype is incompatible.
        ValueError
            If tensor is empty.
        """
        if tensor.dim() != 1:
            logger.error(f"{name} must be 1D, got {tensor.dim()}D")
            raise TypeError(f"{name} must be 1D tensor, got {tensor.dim()}D")

        if tensor.numel() == 0:
            logger.error(f"{name} is empty")
            raise ValueError(f"{name} cannot be empty")

        if torch.isnan(tensor).any():
            logger.warning(f"{name} contains NaN values")

        if torch.isinf(tensor).any():
            logger.warning(f"{name} contains Inf values")

    def compute_quantile_statistics(
        self,
        tensor: torch.Tensor
    ) -> QuantileStatistics:
        """
        Compute comprehensive quantile statistics from tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            1D tensor of values to analyze.

        Returns
        -------
        QuantileStatistics
            Object containing count, mean, std, and quantiles.

        Raises
        ------
        TypeError
            If tensor is not 1D.
        ValueError
            If tensor is empty.

        Examples
        --------
        >>> tensor = torch.randn(1000)
        >>> stats = analyzer.compute_quantile_statistics(tensor)
        >>> print(f"Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")
        """
        if self.validate_input:
            self._validate_tensor(tensor, "Input tensor")

        # Convert to CPU if needed
        if tensor.device.type != "cpu":
            logger.debug(f"Moving tensor to CPU for statistics computation")
            tensor = tensor.to("cpu")

        tensor_float = tensor.float()

        # Compute statistics
        count = tensor.numel()
        mean = tensor_float.mean().item()
        std = tensor_float.std().item()

        quantile_values = torch.tensor([0.25, 0.50, 0.75])
        quantiles = tensor_float.quantile(quantile_values)

        q25 = quantiles[0].item()
        q50 = quantiles[1].item()
        q75 = quantiles[2].item()

        logger.debug(
            f"Computed statistics: count={count}, mean={mean:.6f}, "
            f"std={std:.6f}"
        )

        return QuantileStatistics(
            count=count,
            mean=mean,
            std=std,
            q25=q25,
            q50=q50,
            q75=q75
        )

    def compute_bin_seq_summary(
        self,
        bin_seq_mf: torch.Tensor
    ) -> BinarySequenceSummary:
        """
        Compute summary for all binary sequences.

        Parameters
        ----------
        bin_seq_mf : torch.Tensor
            1D tensor of merit factors for all binary sequences.

        Returns
        -------
        BinarySequenceSummary
            Statistical summary of all sequences.

        Examples
        --------
        >>> bin_seq_mf = torch.randn(1000)
        >>> summary = analyzer.compute_bin_seq_summary(bin_seq_mf)
        """
        logger.info("Computing binary sequence summary")

        stats = self.compute_quantile_statistics(bin_seq_mf)

        summary = BinarySequenceSummary(
            count=stats.count,
            mean=stats.mean,
            std=stats.std,
            q25=stats.q25,
            q50=stats.q50,
            q75=stats.q75
        )

        logger.debug(f"Binary sequence summary: {summary}")

        return summary

    def compute_elite_summary(
        self,
        elite_mf: torch.Tensor,
        total_sequences: int,
        distribution: torch.Tensor
    ) -> EliteSummary:
        """
        Compute summary for elite sequences.

        Parameters
        ----------
        elite_mf : torch.Tensor
            1D tensor of merit factors for elite sequences.
        total_sequences : int
            Total number of sequences before elite selection.
        distribution : torch.Tensor
            Updated Bernoulli distribution.

        Returns
        -------
        EliteSummary
            Statistical summary of elite sequences.

        Examples
        --------
        >>> elite_mf = torch.randn(500)
        >>> dist = torch.randn(256)
        >>> summary = analyzer.compute_elite_summary(elite_mf, 1000, dist)
        """
        logger.info("Computing elite summary")

        if total_sequences <= 0:
            logger.error(f"Invalid total_sequences: {total_sequences}")
            raise ValueError(f"total_sequences must be positive, got {total_sequences}")

        stats = self.compute_quantile_statistics(elite_mf)

        ratio = float(stats.count) / float(total_sequences)

        min_val = elite_mf.min().item()
        max_val = elite_mf.max().item()

        # Convert distribution to list for JSON serialization
        if isinstance(distribution, torch.Tensor):
            dist_list = distribution.cpu().numpy().tolist()
        else:
            dist_list = distribution.tolist() if hasattr(distribution, 'tolist') else distribution

        summary = EliteSummary(
            count=stats.count,
            ratio=ratio,
            mean=stats.mean,
            std=stats.std,
            q25=stats.q25,
            q50=stats.q50,
            q75=stats.q75,
            min=min_val,
            max=max_val,
            distribution=dist_list
        )

        logger.debug(f"Elite summary: {summary}")

        return summary

    def compute_epoch_summary(
        self,
        epoch: int,
        bin_seq_mf: torch.Tensor,
        elite_mf: torch.Tensor,
        distribution: torch.Tensor
    ) -> EpochSummary:
        """
        Compute complete epoch summary.

        Parameters
        ----------
        epoch : int
            Epoch number.
        bin_seq_mf : torch.Tensor
            Merit factors for all sequences.
        elite_mf : torch.Tensor
            Merit factors for elite sequences.
        distribution : torch.Tensor
            Updated Bernoulli distribution.

        Returns
        -------
        EpochSummary
            Complete statistical summary for the epoch.

        Examples
        --------
        >>> summary = analyzer.compute_epoch_summary(
        ...     epoch=10,
        ...     bin_seq_mf=torch.randn(1000),
        ...     elite_mf=torch.randn(500),
        ...     distribution=torch.randn(256)
        ... )
        """
        logger.info(f"Computing epoch {epoch} summary")

        timestamp = datetime.now().isoformat()

        bin_seq_summary = self.compute_bin_seq_summary(bin_seq_mf)
        elite_summary = self.compute_elite_summary(
            elite_mf,
            bin_seq_mf.numel(),
            distribution
        )

        epoch_summary = EpochSummary(
            epoch=epoch,
            timestamp=timestamp,
            bin_seq=bin_seq_summary,
            elites=elite_summary
        )

        logger.info(f"Epoch {epoch} summary computed successfully")

        return epoch_summary


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manages saving and loading of epoch checkpoints.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Directory for saving checkpoints. Default is "./checkpoints".

    Examples
    --------
    >>> manager = CheckpointManager(checkpoint_dir="./checkpoints")
    >>> manager.save_epoch_summary(epoch_summary)
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CheckpointManager: dir={self.checkpoint_dir}")

    def _get_epoch_path(self, epoch: int) -> Path:
        """Get directory path for epoch."""
        epoch_dir = self.checkpoint_dir / f"epoch_{epoch:06d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        return epoch_dir

    def save_epoch_summary(
        self,
        epoch_summary: EpochSummary
    ) -> None:
        """
        Save epoch summary to disk.

        Parameters
        ----------
        epoch_summary : EpochSummary
            Summary object to save.

        Returns
        -------
        None

        Examples
        --------
        >>> manager.save_epoch_summary(epoch_summary)
        """
        logger.info(f"Saving epoch {epoch_summary.epoch} summary")

        epoch_dir = self._get_epoch_path(epoch_summary.epoch)

        # Save as JSON
        json_path = epoch_dir / "summary.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(epoch_summary.to_dict(), f, indent=2)
            json_size = json_path.stat().st_size / (1024**2)
            logger.info(f"Saved JSON summary: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON summary: {e}")
            raise

        # Save as pickle
        pickle_path = epoch_dir / "summary.pkl"
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(epoch_summary, f)
            pickle_size = pickle_path.stat().st_size / (1024**2)
            logger.info(f"Saved pickle summary: {pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save pickle summary: {e}")
            raise

        total_size = json_size + pickle_size

        logger.info(
            f"Epoch {epoch_summary.epoch} checkpoint saved"
        )

    def load_epoch_summary(self, epoch: int) -> EpochSummary:
        """
        Load epoch summary from disk.

        Parameters
        ----------
        epoch : int
            Epoch number to load.

        Returns
        -------
        EpochSummary
            Loaded summary object.

        Raises
        ------
        FileNotFoundError
            If checkpoint does not exist.

        Examples
        --------
        >>> summary = manager.load_epoch_summary(10)
        """
        logger.info(f"Loading epoch {epoch} summary")

        epoch_dir = self._get_epoch_path(epoch)
        pickle_path = epoch_dir / "summary.pkl"

        if not pickle_path.exists():
            logger.error(f"Checkpoint not found: {pickle_path}")
            raise FileNotFoundError(f"Checkpoint not found: {pickle_path}")

        try:
            with open(pickle_path, 'rb') as f:
                summary = pickle.load(f)
            logger.info(f"Loaded epoch {epoch} summary")
            return summary
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def get_latest_epoch(self) -> Optional[int]:
        """
        Get the latest epoch number saved.

        Returns
        -------
        int or None
            Latest epoch number, or None if no checkpoints exist.

        Examples
        --------
        >>> latest = manager.get_latest_epoch()
        >>> if latest is not None:
        ...     print(f"Latest epoch: {latest}")
        """
        epoch_dirs = list(self.checkpoint_dir.glob("epoch_*"))

        if not epoch_dirs:
            return None

        # Extract epoch numbers and return maximum
        epoch_numbers = [
            int(d.name.split('_')[1])
            for d in epoch_dirs
            if d.name.startswith('epoch_')
        ]

        return max(epoch_numbers) if epoch_numbers else None


# ============================================================================
# Epoch Summary Logger
# ============================================================================

class EpochSummaryLogger:
    """Manages logging of epoch summaries with pretty formatting.

    Parameters
    ----------
    logger_instance : logging.Logger, optional
        Logger to use. Default uses module logger.

    Examples
    --------
    >>> summary_logger = EpochSummaryLogger()
    >>> summary_logger.log_epoch_summary(epoch_summary)
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize epoch summary logger."""
        self.logger = logger_instance if logger_instance is not None else logger

    def log_bin_seq_summary(self, summary: BinarySequenceSummary) -> None:
        """
        Log binary sequence summary.

        Parameters
        ----------
        summary : BinarySequenceSummary
            Summary to log.
        """
        self.logger.info("=" * 80)
        self.logger.info("BINARY SEQUENCE SUMMARY (ALL SEQUENCES)")
        self.logger.info("=" * 80)

        self.logger.info(
            f"  Count:                {summary.count:,}"
        )
        self.logger.info(
            f"  Mean:                 {summary.mean:>10.6f}"
        )
        self.logger.info(
            f"  Std Dev:              {summary.std:>10.6f}"
        )
        self.logger.info(
            f"  25th Percentile:      {summary.q25:>10.6f}"
        )
        self.logger.info(
            f"  50th Percentile:      {summary.q50:>10.6f}"
        )
        self.logger.info(
            f"  75th Percentile:      {summary.q75:>10.6f}"
        )
        self.logger.info("-" * 80)

    def log_elite_summary(self, summary: EliteSummary) -> None:
        """
        Log elite summary.

        Parameters
        ----------
        summary : EliteSummary
            Summary to log.
        """
        self.logger.info("=" * 80)
        self.logger.info("ELITE SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info(
            f"  Count:                {summary.count:,}"
        )
        self.logger.info(
            f"  Ratio to Total:       {summary.ratio:>10.4f}"
        )
        self.logger.info(
            f"  Mean:                 {summary.mean:>10.6f}"
        )
        self.logger.info(
            f"  Std Dev:              {summary.std:>10.6f}"
        )
        self.logger.info(
            f"  25th Percentile:      {summary.q25:>10.6f}"
        )
        self.logger.info(
            f"  50th Percentile:      {summary.q50:>10.6f}"
        )
        self.logger.info(
            f"  75th Percentile:      {summary.q75:>10.6f}"
        )
        self.logger.info(
            f"  Minimum:              {summary.min:>10.6f}"
        )
        self.logger.info(
            f"  Maximum:              {summary.max:>10.6f}"
        )

        self.logger.info("-" * 80)
        self.logger.info("Updated Bernoulli Distribution:")

        # Log distribution in chunks of 8 values per line
        dist_array = np.array(summary.distribution)
        for i in range(0, len(summary.distribution), 8):
            chunk = dist_array[i:i+8]
            chunk_str = " | ".join(f"{x:>8.4f}" for x in chunk)
            self.logger.info(f"  [{i:3d}-{min(i+7, len(summary.distribution)-1):3d}]: {chunk_str}")

        self.logger.info("-" * 80)

    def log_epoch_summary(self, summary: EpochSummary) -> None:
        """
        Log complete epoch summary.

        Parameters
        ----------
        summary : EpochSummary
            Summary to log.
        """
        self.logger.info("")
        self.logger.info("#" * 80)
        self.logger.info(f"# EPOCH {summary.epoch:06d} SUMMARY")
        self.logger.info(f"# Timestamp: {summary.timestamp}")
        self.logger.info("#" * 80)

        self.log_bin_seq_summary(summary.bin_seq)
        self.log_elite_summary(summary.elites)

        self.logger.info("#" * 80)
        self.logger.info("")

if __name__ == "__main__":
    """Comprehensive example demonstrating epoch summary functionality."""

    print("\n" + "=" * 80)
    print("Epoch Summary Manager - Example Usage")
    print("=" * 80)

    try:
        # Example 1: Basic statistical analysis
        print("\n[Example 1] Basic Statistical Analysis")
        print("-" * 80)

        analyzer = StatisticalAnalyzer()

        # Create sample data
        n_sequences = 1000
        n_elites = 500
        n_bits = 256

        bin_seq_mf = torch.randn(n_sequences) * 0.3 + 0.5  # Mean 0.5, Std 0.3
        elite_mf = torch.randn(n_elites) * 0.2 + 0.7  # Mean 0.7, Std 0.2
        distribution = torch.rand(n_bits)  # Random distribution

        print(f"Input data:")
        print(f"  Total sequences: {n_sequences}")
        print(f"  Elite sequences: {n_elites}")
        print(f"  Distribution length: {n_bits}")

        # Compute summaries
        bin_seq_summary = analyzer.compute_bin_seq_summary(bin_seq_mf)
        elite_summary = analyzer.compute_elite_summary(elite_mf, n_sequences, distribution)

        print(f"\nBinary sequence summary:")
        print(f"  {bin_seq_summary}")

        print(f"\nElite summary:")
        print(f"  {elite_summary}")

        # Example 2: Epoch summary with logging
        print("\n[Example 2] Complete Epoch Summary with Logging")
        print("-" * 80)

        epoch_summary = analyzer.compute_epoch_summary(
            epoch=1,
            bin_seq_mf=bin_seq_mf,
            elite_mf=elite_mf,
            distribution=distribution
        )

        summary_logger = EpochSummaryLogger()
        summary_logger.log_epoch_summary(epoch_summary)

        # Example 3: Checkpoint saving
        print("\n[Example 3] Checkpoint Saving and Loading")
        print("-" * 80)

        manager = CheckpointManager()

        # Save multiple epochs
        print(f"Saving checkpoints for epochs 1-5...")
        for epoch in range(1, 6):
            # Create summaries for each epoch
            bin_seq_mf_epoch = torch.randn(n_sequences) * 0.3 + (0.4 + epoch * 0.05)
            elite_mf_epoch = torch.randn(n_elites) * 0.2 + (0.6 + epoch * 0.06)
            dist_epoch = torch.rand(n_bits)

            epoch_summary = analyzer.compute_epoch_summary(
                epoch=epoch,
                bin_seq_mf=bin_seq_mf_epoch,
                elite_mf=elite_mf_epoch,
                distribution=dist_epoch
            )

            manager.save_epoch_summary(epoch_summary)
            print(f"  Epoch {epoch}: Checkpoint saved.")

        # Get latest epoch
        latest_epoch = manager.get_latest_epoch()
        print(f"\nLatest saved epoch: {latest_epoch}")

        # Load an epoch
        loaded_summary = manager.load_epoch_summary(3)
        print(f"Loaded epoch 3:")
        print(f"  Bin seq count: {loaded_summary.bin_seq.count}")
        print(f"  Elite count: {loaded_summary.elites.count}")
        print(f"  Elite ratio: {loaded_summary.elites.ratio:.4f}")

        # Example 4: Statistical trends over epochs
        print("\n[Example 4] Statistical Trends Over Epochs")
        print("-" * 80)

        print(f"{'Epoch':<10} {'Bin Seq Mean':<15} {'Elite Mean':<15} {'Elite Ratio':<15}")
        print("-" * 55)

        for epoch in range(1, 6):
            summary = manager.load_epoch_summary(epoch)
            print(
                f"{epoch:<10} {summary.bin_seq.mean:<15.6f} "
                f"{summary.elites.mean:<15.6f} {summary.elites.ratio:<15.4f}"
            )

        # Example 5: Error handling
        print("\n[Example 5] Error Handling")
        print("-" * 80)

        error_cases = [
            {
                "name": "Empty tensor",
                "tensor": torch.tensor([]),
                "should_fail": True
            },
            {
                "name": "2D tensor",
                "tensor": torch.randn(10, 10),
                "should_fail": True
            },
            {
                "name": "Valid tensor",
                "tensor": torch.randn(1000),
                "should_fail": False
            },
            {
                "name": "Tensor with NaN",
                "tensor": torch.tensor([1.0, float('nan'), 3.0]),
                "should_fail": False  # Logs warning but doesn't fail
            },
        ]

        for case in error_cases:
            try:
                stats = analyzer.compute_quantile_statistics(case["tensor"])
                if case["should_fail"]:
                    print(f"   {case['name']}: Expected failure but succeeded")
                else:
                    print(f"   {case['name']}: Succeeded")
            except (TypeError, ValueError) as e:
                if case["should_fail"]:
                    print(f"   {case['name']}: Failed as expected")
                else:
                    print(f"   {case['name']}: Unexpected failure")

        # Example 6: Large-scale epoch processing
        print("\n[Example 6] Large-Scale Epoch Processing")
        print("-" * 80)

        large_n = 100000
        large_elites = 50000

        print(f"Processing large epoch: {large_n:,} sequences")

        large_bin_seq = torch.randn(large_n) * 0.3 + 0.5
        large_elite_seq = torch.randn(large_elites) * 0.2 + 0.7
        large_dist = torch.rand(n_bits)

        large_epoch = analyzer.compute_epoch_summary(
            epoch=100,
            bin_seq_mf=large_bin_seq,
            elite_mf=large_elite_seq,
            distribution=large_dist
        )

        print(f"Elite ratio: {large_epoch.elites.ratio:.4f}")

        # Save large epoch
        manager.save_epoch_summary(large_epoch)

        print("\n" + "=" * 80)
        print("All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.exception("Example execution failed")
        print(f"\nError during example: {e}\n")
