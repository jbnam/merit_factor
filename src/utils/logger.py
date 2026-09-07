"""
Experiment Logger - Unified Training and Monitoring System
===========================================================

An experiment logging system integrating:
- Real-time resource monitoring (CPU, RAM, GPU VRAM)
- Training metrics tracking and persistence
- Statistical summaries and checkpointing
- HDF5 data management
- Performance profiling
- Comprehensive error handling

This module provides a unified interface for all experiment tracking needs
in machine learning training pipelines.

Author: NAMRI
License: Apache License 2.0
Version: 0.1.0

Example
-------
    >>> from src.utils.logger import ExperimentLogger
    >>> logger = ExperimentLogger("cem_golay", len_bin_seq=256)
    >>> logger.log_initialization(config_dict)
    >>> for epoch in range(num_epochs):
    ...     with logger.log_epoch_context(epoch, samples=10000) as metrics:
    ...         metrics["loss"] = compute_loss()
    ...         metrics["accuracy"] = compute_accuracy()
    >>> logger.log_summary()
    >>> logger.save_all_metrics()
"""

import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Generator
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

import psutil
import torch

# ============================================================================
# File Names and Environment Constants
# ============================================================================

# File names
LOG_FILE_NAME = "experiment.log"

# Environment variable for log directory
DIR_PROJECT=str(Path(__file__).resolve().parent.parent.parent)
os.environ["DIR_PROJECT"] = DIR_PROJECT

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class LoggerConfig:
    """Configuration for experiment logger.

    Parameters
    ----------
    log_dir : str, optional
        Base directory for logs. Default is "./logs".
    resource_cache_interval : float, optional
        Resource monitoring cache interval in seconds. Default is 1.0.
    max_log_file_size_mb : int, optional
        Maximum log file size before rotation in MB. Default is 500.
    backup_count : int, optional
        Number of backup log files to keep. Default is 100.
    enable_resource_monitoring : bool, optional
        Enable resource monitoring. Default is True.

    Raises
    ------
    ValueError
        If configuration values are invalid.
    """
    log_dir: str = os.path.join(os.getenv("DIR_PROJECT", "."), "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    resource_cache_interval: float = 1.0
    max_log_file_size_mb: int = 500
    backup_count: int = 10
    enable_resource_monitoring: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.resource_cache_interval < 0:
            raise ValueError(f"resource_cache_interval must be >= 0, got {self.resource_cache_interval}")

        if self.max_log_file_size_mb <= 0:
            raise ValueError(f"max_log_file_size_mb must be > 0, got {self.max_log_file_size_mb}")

        if self.backup_count < 0:
            raise ValueError(f"backup_count must be >= 0, got {self.backup_count}")


# ============================================================================
# Resource Monitoring
# ============================================================================

@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage.

    Attributes
    ----------
    timestamp : str
        ISO format timestamp.
    uptime : str
        Formatted uptime duration.
    cpu_percent : float
        CPU utilization percentage (0-100).
    ram_used_gb : float
        Used system RAM in GB.
    ram_total_gb : float
        Total system RAM in GB.
    gpu_allocated_gb : float
        Allocated GPU memory in GB.
    gpu_reserved_gb : float
        Reserved GPU memory in GB.
    """
    timestamp: str
    uptime: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float

    def __str__(self) -> str:
        """Return formatted string representation."""
        return (
            f"[{self.uptime}] [CPU: {self.cpu_percent:.1f}%] "
            f"[RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f}GB] "
            f"[GPU: {self.gpu_allocated_gb:.2f}/{self.gpu_reserved_gb:.2f}GB]"
        )


class ResourceMonitor:
    """Monitor system resource usage with caching optimization.

    Parameters
    ----------
    cache_interval_s : float, optional
        Cache duration in seconds. Default is 1.0.

    Attributes
    ----------
    cache_interval_s : float
        Cache interval.
    start_time : float
        Process start time.
    process : psutil.Process
        Process object for resource tracking.
    last_update : float
        Timestamp of last cache update.
    cached_snapshot : ResourceSnapshot
        Cached resource snapshot.

    Notes
    -----
    Resource monitoring uses caching to minimize overhead.
    Cached data is reused within the cache interval.
    """

    def __init__(self, cache_interval_s: float = 1.0):
        """Initialize resource monitor."""
        self.cache_interval_s = cache_interval_s
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.last_update = 0.0
        self.cached_snapshot: Optional[ResourceSnapshot] = None
        self._lock = threading.Lock()

    def get_snapshot(self, force_update: bool = False) -> ResourceSnapshot:
        """Get current resource snapshot.

        Parameters
        ----------
        force_update : bool, optional
            Force update even if cached. Default is False.

        Returns
        -------
        ResourceSnapshot
            Current resource snapshot.

        Notes
        -----
        Returns cached snapshot if cache is still valid and force_update is False.
        """
        current_time = time.time()

        # Return cached if valid
        if (
            not force_update
            and self.cached_snapshot is not None
            and (current_time - self.last_update) < self.cache_interval_s
        ):
            return self.cached_snapshot

        with self._lock:
            elapsed = current_time - self.start_time
            uptime_str = str(timedelta(seconds=int(elapsed)))

            # CPU metrics with error handling
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
            except Exception:
                cpu_percent = 0.0

            # RAM metrics with error handling
            try:
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024 ** 3)
                ram_total_gb = ram.total / (1024 ** 3)
            except Exception:
                ram_used_gb = 0.0
                ram_total_gb = 0.0

            # GPU metrics with error handling
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0

            if torch.cuda.is_available():
                try:
                    # WARNING! torch.cuda operations may raise RuntimeError
                    gpu_id = torch.cuda.current_device()
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    reserved = torch.cuda.memory_reserved(gpu_id)

                    gpu_allocated_gb = allocated / (1024 ** 3)
                    gpu_reserved_gb = reserved / (1024 ** 3)
                except RuntimeError:
                    # GPU operations failed gracefully
                    pass

            self.cached_snapshot = ResourceSnapshot(
                timestamp=datetime.now().isoformat(),
                uptime=uptime_str,
                cpu_percent=cpu_percent,
                ram_used_gb=ram_used_gb,
                ram_total_gb=ram_total_gb,
                gpu_allocated_gb=gpu_allocated_gb,
                gpu_reserved_gb=gpu_reserved_gb
            )

            self.last_update = current_time
            return self.cached_snapshot

    def reset_uptime(self) -> None:
        """Reset uptime counter."""
        self.start_time = time.time()
        self.last_update = 0.0


# ============================================================================
# Custom Formatter
# ============================================================================

class ExperimentFormatter(logging.Formatter):
    """Custom formatter with resource metrics.

    Parameters
    ----------
    fmt : str, optional
        Log format string.
    datefmt : str, optional
        Date format string.
    monitor : ResourceMonitor, optional
        Resource monitor instance.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize formatter."""
        super().__init__(fmt, datefmt)
        self.monitor = monitor if monitor is not None else ResourceMonitor()

    def format(self, record: logging.LogRecord) -> str:
        """Format record with resource metrics.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            Formatted log message.

        WARNING! Accessing monitor may block briefly during resource queries
        """
        try:
            snapshot = self.monitor.get_snapshot()

            record.uptime = snapshot.uptime
            record.cpu_pct = f"{snapshot.cpu_percent:.1f}%"
            record.ram_gb = f"{snapshot.ram_used_gb:.1f}/{snapshot.ram_total_gb:.1f}GB"
            record.gpu_gb = f"{snapshot.gpu_allocated_gb:.2f}/{snapshot.gpu_reserved_gb:.2f}GB"
        except Exception:
            # Fallback if resource monitoring fails
            record.uptime = "N/A"
            record.cpu_pct = "N/A"
            record.ram_gb = "N/A"
            record.gpu_gb = "N/A"

        return super().format(record)


# ============================================================================
# Metrics Data Classes
# ============================================================================

@dataclass
class EpochMetrics:
    """Metrics for a single epoch.

    Attributes
    ----------
    epoch : int
        Epoch number.
    timestamp : str
        ISO format timestamp.
    metrics : dict
        Dictionary of metric values.
    samples_processed : int
        Number of samples processed.
    execution_time_s : float
        Execution time in seconds.
    throughput : float
        Throughput (samples/second).
    """
    epoch: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    samples_processed: int = 0
    execution_time_s: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentSummary:
    """Summary of experiment execution.

    Attributes
    ----------
    total_epochs : int
        Total number of epochs.
    total_samples : int
        Total samples processed.
    total_time_s : float
        Total execution time in seconds.
    epoch_metrics : list
        List of epoch metrics.
    """
    total_epochs: int = 0
    total_samples: int = 0
    total_time_s: float = 0.0
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)


# ============================================================================
# Main Experiment Logger
# ============================================================================

class ExperimentLogger:
    """Experiment logger with comprehensive tracking.

    Integrates resource monitoring, training metrics, data persistence,
    and statistical summaries for complete experiment tracking.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    len_bin_seq : int, optional
        Binary sequence length for Golay problem (used in path).
    config : LoggerConfig, optional
        Logger configuration.

    Attributes
    ----------
    experiment_name : str
        Experiment name.
    log_dir : Path
        Logging directory.
    logger : logging.Logger
        Main logger instance.
    resource_monitor : ResourceMonitor
        Resource monitoring instance.
    summary : ExperimentSummary
        Experiment summary data.

    Examples
    --------
    >>> logger = ExperimentLogger("cem_golay", len_bin_seq=256)
    >>> logger.log_initialization({"num_epochs": 10})
    >>> for epoch in range(10):
    ...     with logger.log_epoch_context(epoch, samples=1000) as metrics:
    ...         metrics["loss"] = 0.5
    >>> logger.log_summary()
    >>> logger.save_all_metrics()
    """

    def __init__(
        self,
        experiment_name: str,
        len_bin_seq: Optional[int] = None,
        config: Optional[LoggerConfig] = None
    ):
        """Initialize experiment logger.

        Parameters
        ----------
        experiment_name : str
            Name of experiment.
        len_bin_seq : int, optional
            Binary sequence length.
        config : LoggerConfig, optional
            Logger configuration.
        """
        self.experiment_name = experiment_name
        self.len_bin_seq = len_bin_seq
        self.config = config if config is not None else LoggerConfig()
        self.start_time = time.time()

        # Setup directories
        self._setup_directories()

        # Setup monitoring
        self.resource_monitor = ResourceMonitor(self.config.resource_cache_interval)
        self.summary = ExperimentSummary()

        # Setup logger
        self.logger = self._setup_logger()

        self.logger.info(f"ExperimentLogger initialized: {experiment_name}")

    def _setup_directories(self) -> None:
        """Setup log directory structure.
        """
        base = Path(self.config.log_dir)

        if self.len_bin_seq:
            self.log_dir = base / self.experiment_name / str(self.len_bin_seq) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log_dir = base / self.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging system.

        Returns
        -------
        logging.Logger
            Configured logger.
        """
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler
        log_file = self.log_dir / LOG_FILE_NAME

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_log_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
        except OSError as e:
            print(f"Warning: Failed to create log file {log_file}: {e}")
            file_handler = None

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        log_format = (
            "[%(asctime)s] [%(uptime)s] "
            "[CPU: %(cpu_pct)s | RAM: %(ram_gb)s | GPU: %(gpu_gb)s] "
            "[%(levelname)s] %(message)s"
        )
        date_format = "%H:%M:%S"

        formatter = ExperimentFormatter(
            fmt=log_format,
            datefmt=date_format,
            monitor=self.resource_monitor
        )

        if file_handler:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False

        return logger

    def log_initialization(self, config_dict: Dict[str, Any]) -> None:
        """Log experiment initialization and configuration.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary.

        Examples
        --------
        >>> logger.log_initialization({
        ...     "len_bin_seq": 256,
        ...     "num_epochs": 10,
        ...     "method": "naive"
        ... })
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("=" * 80)
        self.logger.info("Configuration:")

        for key, value in sorted(config_dict.items()):
            try:
                self.logger.info(f"  {key}: {value}")
            except Exception as e:
                self.logger.warning(f"  {key}: <non-serializable> ({type(value).__name__})")

        self.logger.info("=" * 80)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        samples_processed: Optional[int] = None,
        execution_time_s: Optional[float] = None
    ) -> None:
        """Log epoch summary.

        Parameters
        ----------
        epoch : int
            Epoch number.
        metrics : dict
            Dictionary of metric values.
        samples_processed : int, optional
            Number of samples processed.
        execution_time_s : float, optional
            Execution time in seconds.

        Examples
        --------
        >>> logger.log_epoch(
        ...     epoch=1,
        ...     metrics={"loss": 0.5, "accuracy": 0.95},
        ...     samples_processed=10000,
        ...     execution_time_s=2.5
        ... )
        """
        if execution_time_s is None:
            execution_time_s = 0.0

        if samples_processed is None:
            samples_processed = 0

        # Format metrics string
        try:
            metrics_str = " | ".join(
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in sorted(metrics.items())
            )
        except Exception:
            metrics_str = str(metrics)

        # Compute throughput safely
        throughput = samples_processed / execution_time_s if execution_time_s > 0 else 0.0

        # Build log message
        msg = f"Epoch {epoch:06d}: {metrics_str}"
        if samples_processed > 0:
            msg += f" | Samples: {samples_processed:,}"
        if execution_time_s > 0:
            msg += f" | Time: {execution_time_s:.3f}s | Throughput: {throughput:.0f} seq/s"

        self.logger.info(msg)

        # Record metrics
        epoch_metric = EpochMetrics(
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            metrics=metrics.copy(),
            samples_processed=samples_processed,
            execution_time_s=execution_time_s,
            throughput=throughput
        )
        self.summary.epoch_metrics.append(epoch_metric)

    def log_batch(
        self,
        epoch: int,
        batch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log batch metrics (debug level).

        Parameters
        ----------
        epoch : int
            Epoch number.
        batch : int
            Batch number.
        metrics : dict
            Batch metric values.

        Examples
        --------
        >>> logger.log_batch(epoch=1, batch=10, metrics={"loss": 0.5})
        """
        try:
            metrics_str = " | ".join(
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in sorted(metrics.items())
            )
        except Exception:
            metrics_str = str(metrics)

        self.logger.debug(f"Epoch {epoch:06d} Batch {batch:06d}: {metrics_str}")

    def log_checkpoint(
        self,
        epoch: int,
        checkpoint_path: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Log checkpoint save.

        Parameters
        ----------
        epoch : int
            Epoch number.
        checkpoint_path : str
            Path to checkpoint file.
        metrics : dict
            Checkpoint metrics.

        Examples
        --------
        >>> logger.log_checkpoint(
        ...     epoch=10,
        ...     checkpoint_path="model_epoch_10.pt",
        ...     metrics={"best_loss": 0.3}
        ... )
        """
        try:
            metrics_str = " | ".join(
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in sorted(metrics.items())
            )
        except Exception:
            metrics_str = str(metrics)

        self.logger.info(
            f"Checkpoint saved (epoch {epoch}): {checkpoint_path} | {metrics_str}"
        )

    @contextmanager
    def log_epoch_context(
        self,
        epoch: int,
        samples: Optional[int] = None
    ) -> Generator[Dict[str, float], None, None]:
        """Context manager for epoch logging.

        Parameters
        ----------
        epoch : int
            Epoch number.
        samples : int, optional
            Number of samples processed.

        Yields
        ------
        dict
            Dictionary to populate with metrics.

        Examples
        --------
        >>> with logger.log_epoch_context(epoch=1, samples=10000) as metrics:
        ...     metrics["loss"] = compute_loss()
        ...     metrics["accuracy"] = compute_accuracy()
        """
        metrics = {}
        start_time = time.time()

        try:
            yield metrics
        finally:
            execution_time = time.time() - start_time
            self.log_epoch(epoch, metrics, samples, execution_time)

    def log_summary(self) -> None:
        """Log experiment summary.

        Examples
        --------
        >>> logger.log_summary()
        """
        total_time = time.time() - self.start_time
        total_samples = sum(
            m.samples_processed for m in self.summary.epoch_metrics
        )

        self.logger.info("=" * 80)
        self.logger.info("Experiment Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Total epochs: {len(self.summary.epoch_metrics)}")
        self.logger.info(f"Total samples: {total_samples:,}")
        self.logger.info(f"Total time: {timedelta(seconds=int(total_time))}")

        if self.summary.epoch_metrics:
            avg_time = total_time / len(self.summary.epoch_metrics)
            self.logger.info(f"Average time/epoch: {avg_time:.3f}s")

            best_epoch = max(
                self.summary.epoch_metrics,
                key=lambda m: m.throughput
            )
            self.logger.info(f"Best throughput: {best_epoch.throughput:.0f} seq/s (epoch {best_epoch.epoch})")

        self.logger.info("=" * 80)

    def save_all_metrics(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """Save all metrics to files.

        Parameters
        ----------
        output_dir : str, optional
            Output directory. Uses log_dir if None.

        Returns
        -------
        tuple of (str, str)
            Paths to (metrics_file, summary_file).

        Raises
        ------
        IOError
            If file I/O fails.

        Examples
        --------
        >>> metrics_path, summary_path = logger.save_all_metrics()
        """
        if output_dir is None:
            output_dir = str(self.log_dir)
        else:
            output_dir = str(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("Experiment Logger - Complete Example")
    print("=" * 80)

    try:
        # Example 1: Initialize logger
        print("\n[Example 1] Logger Initialization")
        print("-" * 80)


        logger = ExperimentLogger("cem_golay", len_bin_seq=1024)

        logger.log_initialization({
            "len_bin_seq": 1024,
            "method_dist": "naive",
            "num_epochs": 5,
            "num_samples": 1000,
            "ratio": 0.5,
            "block_size": 256,
            "num_blocks": 4
        })

        print(f"  Logger initialized")
        print(f"  Log directory: {logger.log_dir}")

        # Example 2: Training simulation
        print("\n[Example 2] Training Simulation")
        print("-" * 80)

        num_epochs = 5
        for epoch in range(1, num_epochs + 1):
            # Simulate batches
            for batch in range(10):
                time.sleep(0.01)

                batch_metrics = {
                    "loss": 0.5 - epoch * 0.05 - batch * 0.01,
                    "accuracy": 0.5 + epoch * 0.05 + batch * 0.01
                }
                logger.log_batch(epoch, batch, batch_metrics)

            # Log epoch with context manager
            with logger.log_epoch_context(epoch, samples=1000) as metrics:
                time.sleep(0.1)
                metrics["loss"] = 0.5 - epoch * 0.05
                metrics["accuracy"] = 0.5 + epoch * 0.05
                metrics["val_loss"] = 0.55 - epoch * 0.04

            # Checkpoint every 2 epochs
            if epoch % 2 == 0:
                logger.log_checkpoint(
                    epoch=epoch,
                    checkpoint_path=f"model_epoch_{epoch}.pt",
                    metrics={
                        "best_loss": 0.5 - epoch * 0.05,
                        "best_accuracy": 0.5 + epoch * 0.05
                    }
                )

        print(f"✓ Training simulation completed")

        # Example 3: Summary
        print("\n[Example 3] Experiment Summary")
        print("-" * 80)

        logger.log_summary()

        print("\n" + "=" * 80)
        print("All Examples Completed Successfully")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError during example: {e}")
        traceback.print_exc()
        print()
